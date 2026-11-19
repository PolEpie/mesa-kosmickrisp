/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_queue.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_debug.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_cmd_queue.h"

struct kk_commit_data {
   struct kk_queue *queue;
   struct kk_cmd_buffer *cmd;
};

static void
kk_queue_wait_for_commits(struct kk_queue *queue)
{
   mtx_lock(&queue->mutex);
   while (queue->commits_in_flight > 0u ||
          queue->merge_commits_in_flight > 0u)
      cnd_wait(&queue->cond, &queue->mutex);
   mtx_unlock(&queue->mutex);
}

static void
kk_queue_release_cmd_buffer_locked(struct kk_queue *queue,
                                   struct kk_cmd_buffer *cmd)
{
   kk_cmd_buffer_ops.reset(&cmd->vk, 0u);
   util_dynarray_append(&queue->free_cmd_buffers, &cmd->vk);
}

static void
check_device_lost(struct kk_device *dev, struct mtl_feedback_data *data)
{
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      vk_device_set_lost(
         &dev->vk, "Command queue error: %s, with message \"%s\"",
         mtl_command_queue_error_to_string(data->error), data->error_message);
   }
}

static void
commit_callback(struct mtl_feedback_data *data)
{
   check_device_lost((struct kk_device *)data->user_data, data);
}

static void
rerecord_commit_callback(struct mtl_feedback_data *data)
{
   struct kk_commit_data *commit = (struct kk_commit_data *)data->user_data;
   struct kk_queue *queue = commit->queue;
   struct kk_device *dev = kk_queue_device(queue);

   check_device_lost(dev, data);

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   kk_queue_release_cmd_buffer_locked(queue, commit->cmd);

   assert(queue->commits_in_flight > 0u);
   if (--queue->commits_in_flight == 0u) {
      kk_ResetCommandPool(kk_device_to_handle(dev),
                          kk_cmd_pool_to_handle(queue->cmd_pool), 0u);
      cnd_broadcast(&queue->cond);
   }

   mtx_unlock(&queue->mutex);

   vk_free(&dev->vk.alloc, commit);
}

static bool
kk_cmd_buffer_has_work(struct kk_cmd_buffer *cmd)
{
   return cmd->metal.cmd_buf;
}

struct kk_merge_commit {
   struct kk_queue *queue;
   struct util_dynarray bookends;
};

static void
merge_commit_callback(struct mtl_feedback_data *data)
{
   struct kk_merge_commit *merge = (struct kk_merge_commit *)data->user_data;
   struct kk_queue *queue = merge->queue;
   struct kk_device *dev = kk_queue_device(queue);

   mtx_lock(&queue->mutex);
   util_dynarray_append_dynarray(&queue->merge_bookends, &merge->bookends);
   if (--queue->merge_commits_in_flight == 0u)
      cnd_broadcast(&queue->cond);
   mtx_unlock(&queue->mutex);

   util_dynarray_fini(&merge->bookends);
   vk_free(&dev->vk.alloc, merge);
}

static struct kk_merge_commit *
kk_queue_begin_merge(struct kk_queue *queue)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct kk_merge_commit *merge = vk_alloc(
      &dev->vk.alloc, sizeof(*merge), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (merge == NULL)
      return NULL;

   merge->queue = queue;
   merge->bookends = UTIL_DYNARRAY_INIT;

   mtx_lock(&queue->mutex);
   queue->merge_commits_in_flight++;
   mtx_unlock(&queue->mutex);

   return merge;
}

static mtl_command_buffer *
kk_queue_begin_bookend(struct kk_queue *queue, struct kk_merge_commit *merge)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct kk_bookend bookend = {0};

   mtx_lock(&queue->mutex);
   if (util_dynarray_num_elements(&queue->merge_bookends, struct kk_bookend) >
       0u)
      bookend = util_dynarray_pop(&queue->merge_bookends, struct kk_bookend);
   mtx_unlock(&queue->mutex);
   if (bookend.cmd_buf == NULL) {
      bookend.cmd_buf = mtl_new_command_buffer(dev->mtl_handle);
      bookend.allocator = mtl_new_command_allocator(dev->mtl_handle);
   } else {
      mtl_command_allocator_reset(bookend.allocator);
   }

   mtl_begin_command_buffer(bookend.cmd_buf, bookend.allocator);
   util_dynarray_append(&merge->bookends, bookend);
   return bookend.cmd_buf;
}

static mtl_command_buffer *
kk_queue_bookend(struct kk_queue *queue, struct kk_merge_commit *merge,
                 mtl_render_pass_descriptor *descriptor, bool suspending)
{
   mtl_command_buffer *cmd_buf = kk_queue_begin_bookend(queue, merge);
   mtl_render_encoder *encoder = mtl_new_render_command_encoder_with_options(
      cmd_buf, descriptor, suspending, !suspending);
   if (suspending)
      mtl_wait_for_fence(encoder, queue->merge_fence, MTL_STAGE_VERTEX);
   else
      mtl_barrier_after_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_end_encoding(encoder);
   mtl_release(encoder);
   mtl_end_command_buffer(cmd_buf);
   return cmd_buf;
}

/* Vertex work recorded in resuming render encoders is not held back by queue
 * barriers: neither by the producer barriers ending earlier encoders nor by
 * consumer barriers in the merged pass itself. Fragment work is. Fences are
 * honored, so wait here for the earlier stages that vertex work recorded since
 * the last wait depends on, and update the fence the opener waits on before
 * its vertex stage. */
static mtl_command_buffer *
kk_queue_merge_wait(struct kk_queue *queue, struct kk_merge_commit *merge,
                    enum mtl_stages stages)
{
   mtl_command_buffer *cmd_buf = kk_queue_begin_bookend(queue, merge);
   mtl_compute_encoder *encoder = mtl_new_compute_command_encoder(cmd_buf);
   mtl_barrier_after_queue_stages(encoder, stages, MTL_STAGE_DISPATCH);
   mtl_update_fence(encoder, queue->merge_fence, MTL_STAGE_DISPATCH);
   mtl_end_encoding(encoder);
   mtl_release(encoder);
   mtl_end_command_buffer(cmd_buf);
   return cmd_buf;
}

static mtl_command_buffer *
kk_queue_merge_fence(struct kk_queue *queue, struct kk_merge_commit *merge)
{
   mtl_command_buffer *cmd_buf = kk_queue_begin_bookend(queue, merge);
   mtl_compute_encoder *encoder = mtl_new_compute_command_encoder(cmd_buf);
   mtl_barrier_after_queue_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_barrier_after_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_end_encoding(encoder);
   mtl_release(encoder);
   mtl_end_command_buffer(cmd_buf);
   return cmd_buf;
}

static bool
kk_segments_chain(const struct kk_cmd_segment *a,
                  const struct kk_cmd_segment *b)
{
   return a->merge_desc && b->merge_desc && !a->break_after &&
          !b->break_before && memcmp(&a->key, &b->key, sizeof(a->key)) == 0;
}

static void
kk_queue_commit_group(struct kk_queue *queue, struct util_dynarray *cmd_bufs,
                      mtl_feedback_handler_callback callback, void *user_data,
                      struct kk_merge_commit *merge)
{
   mtl_commit_options *options = mtl_new_commit_options();
   if (callback)
      mtl_commit_options_add_feedback_handler(options, callback, user_data);
   if (merge)
      mtl_commit_options_add_feedback_handler(options, merge_commit_callback,
                                              merge);
   mtl_command_queue_commit(
      queue->mtl_handle, util_dynarray_begin(cmd_bufs),
      util_dynarray_num_elements(cmd_bufs, mtl_command_buffer *), options);
   mtl_release(options);
   util_dynarray_clear(cmd_bufs);
}

static void
kk_queue_commit_segments(struct kk_queue *queue,
                         const struct kk_cmd_segment *segments, uint32_t count,
                         mtl_feedback_handler_callback callback,
                         void *user_data)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct util_dynarray cmd_bufs = UTIL_DYNARRAY_INIT;
   struct kk_merge_commit *merge = NULL;

   for (uint32_t i = 0; i < count; i++) {
      const struct kk_cmd_segment *segment = &segments[i];
      if (segment->merge_desc && merge == NULL) {
         merge = kk_queue_begin_merge(queue);
         if (merge == NULL) {
            vk_device_set_lost(&dev->vk, "Out of memory merging render passes");
            util_dynarray_fini(&cmd_bufs);
            return;
         }
      }

      const bool chain_prev =
         i > 0u && kk_segments_chain(&segments[i - 1u], segment);
      const bool chain_next =
         i + 1u < count && kk_segments_chain(segment, &segments[i + 1u]);

      if (segment->merge_desc && !chain_prev) {
         if (util_dynarray_num_elements(&cmd_bufs, mtl_command_buffer *))
            kk_queue_commit_group(queue, &cmd_bufs, NULL, NULL, NULL);
         if (queue->vertex_wait) {
            util_dynarray_append(
               &cmd_bufs, kk_queue_merge_wait(queue, merge, queue->vertex_wait));
            queue->vertex_wait = 0;
         }
         mtl_command_buffer *opener =
            kk_queue_bookend(queue, merge, segment->merge_desc, true);
         util_dynarray_append(&cmd_bufs, opener);
      }
      util_dynarray_append(&cmd_bufs, segment->cmd_buf);
      queue->vertex_wait |= segment->vertex_wait;
      if (segment->merge_desc && !chain_next) {
         mtl_command_buffer *closer =
            kk_queue_bookend(queue, merge, segment->merge_desc, false);
         util_dynarray_append(&cmd_bufs, closer);
         util_dynarray_append(&cmd_bufs, kk_queue_merge_fence(queue, merge));
         if (i + 1u < count)
            kk_queue_commit_group(queue, &cmd_bufs, NULL, NULL, NULL);
      }
   }

   kk_queue_commit_group(queue, &cmd_bufs, callback, user_data, merge);
   util_dynarray_fini(&cmd_bufs);
}

static void
kk_cmd_buffer_append_segments(struct kk_cmd_buffer *cmd,
                              struct util_dynarray *segments)
{
   struct kk_cmd_segment last = cmd->metal.segment;
   last.cmd_buf = cmd->metal.cmd_buf;

   util_dynarray_append_dynarray(segments, &cmd->metal.segments);
   util_dynarray_append(segments, last);
}

static void
kk_queue_commit(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                mtl_feedback_handler_callback callback, void *user_data)
{
   assert(kk_cmd_buffer_has_work(cmd));

   struct util_dynarray segments = UTIL_DYNARRAY_INIT;
   kk_cmd_buffer_append_segments(cmd, &segments);
   kk_queue_commit_segments(
      queue, util_dynarray_begin(&segments),
      util_dynarray_num_elements(&segments, struct kk_cmd_segment), callback,
      user_data);
   util_dynarray_fini(&segments);
}

static void
kk_queue_flush_batch(struct kk_queue *queue, struct util_dynarray *batch)
{
   const uint32_t count =
      util_dynarray_num_elements(batch, struct kk_cmd_segment);
   if (count == 0u)
      return;

   kk_queue_commit_segments(queue, util_dynarray_begin(batch), count,
                            commit_callback, kk_queue_device(queue));
   util_dynarray_clear(batch);
}

static VkResult
rerecord_and_commit_cmd_buffer(struct kk_queue *queue,
                               struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct vk_command_buffer *vk_cmd = NULL;
   struct kk_cmd_buffer *rerecord;
   VkResult result = VK_SUCCESS;

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   if (util_dynarray_num_elements(&queue->free_cmd_buffers,
                                  struct vk_command_buffer *) > 0u) {
      vk_cmd = util_dynarray_pop(&queue->free_cmd_buffers,
                                 struct vk_command_buffer *);
   } else {
      result = kk_cmd_buffer_ops.create(
         &queue->cmd_pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &vk_cmd);
      if (result != VK_SUCCESS)
         goto unlock;
   }

   rerecord = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VkCommandBuffer rerecord_handle = kk_cmd_buffer_to_handle(rerecord);
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   result = kk_BeginCommandBuffer(rerecord_handle, &begin_info);
   if (result != VK_SUCCESS)
      goto release;

   vk_cmd_queue_execute(&cmd->vk.cmd_queue, rerecord_handle,
                        &dev->vk.dispatch_table);

   result = kk_EndCommandBuffer(rerecord_handle);
   if (result != VK_SUCCESS)
      goto release;

   if (!kk_cmd_buffer_has_work(rerecord))
      goto release;

   struct kk_commit_data *commit = vk_alloc(&dev->vk.alloc, sizeof(*commit), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (commit == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto release;
   }
   commit->queue = queue;
   commit->cmd = rerecord;

   /* Need to ensure the new buffers allocated at record are resident. */
   kk_device_make_resources_resident(dev);

   queue->commits_in_flight++;
   mtx_unlock(&queue->mutex);

   kk_queue_commit(queue, rerecord, rerecord_commit_callback, commit);
   return VK_SUCCESS;

release:
   kk_queue_release_cmd_buffer_locked(queue, rerecord);
unlock:
   mtx_unlock(&queue->mutex);
   return result;
}

static VkResult
kk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct kk_queue *queue = container_of(vk_queue, struct kk_queue, vk);
   struct kk_device *dev = kk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   for (struct vk_sync_wait *wait = submit->waits,
                            *end = submit->waits + submit->wait_count;
        wait != end; ++wait) {
      struct kk_sync_timeline *sync =
         container_of(wait->sync, struct kk_sync_timeline, base);
      mtl_wait_for_event(queue->mtl_handle, sync->mtl_handle, wait->wait_value);
      queue->vertex_wait = MTL_STAGE_ALL;
   }

   /* Ensure any changes to residency are propagated before we submit any
    * work. All resources should have been allocated before submission.
    * Otherwise, users are playing with fire. */
   kk_device_make_resources_resident(dev);

   struct util_dynarray batch = UTIL_DYNARRAY_INIT;
   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd_buffer =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);
      const bool standalone = cmd_buffer->drawable || cmd_buffer->submitted ||
                              KK_DEBUG(NO_MERGE_PASSES);

      if (standalone)
         kk_queue_flush_batch(queue, &batch);

      if (cmd_buffer->drawable) {
         mtl_command_queue_wait_for_drawable(queue->mtl_handle,
                                             cmd_buffer->drawable);
      }

      /* Metal's command buffers are one time use, re-record multiple
       * submissions. */
      if (cmd_buffer->submitted) {
         VkResult result = rerecord_and_commit_cmd_buffer(queue, cmd_buffer);
         if (result != VK_SUCCESS) {
            util_dynarray_fini(&batch);
            return result;
         }
      } else if (kk_cmd_buffer_has_work(cmd_buffer)) {
         if (standalone)
            kk_queue_commit(queue, cmd_buffer, commit_callback, dev);
         else
            kk_cmd_buffer_append_segments(cmd_buffer, &batch);
      }

      cmd_buffer->submitted = true;

      if (cmd_buffer->drawable) {
         mtl_command_queue_signal_drawable(queue->mtl_handle,
                                           cmd_buffer->drawable);
         mtl_release(cmd_buffer->drawable);
         cmd_buffer->drawable = NULL;
      }
   }
   kk_queue_flush_batch(queue, &batch);
   util_dynarray_fini(&batch);

   for (uint32_t i = 0u; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];
      struct kk_sync_timeline *sync =
         container_of(signal->sync, struct kk_sync_timeline, base);
      mtl_signal_event(queue->mtl_handle, sync->mtl_handle,
                       signal->signal_value);
   }

   return VK_SUCCESS;
}

VkResult
kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
              const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   const VkDeviceQueueGlobalPriorityCreateInfo *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
   const VkQueueGlobalPriority global_priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   /* From the Vulkan 1.3.295 spec:
    *
    *    "If the globalPriorityQuery feature is enabled and the requested
    *    global priority is not reported via
    *    VkQueueFamilyGlobalPriorityPropertiesKHR, the driver implementation
    *    must fail the queue creation. In this scenario,
    *    VK_ERROR_INITIALIZATION_FAILED is returned."
    */
   if (dev->vk.enabled_features.globalPriorityQuery &&
       global_priority != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (global_priority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_NOT_PERMITTED;

   VkResult result;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pCreateInfo->queueFamilyIndex,
   };
   VkCommandPool pool_handle;
   result = kk_CreateCommandPool(kk_device_to_handle(dev), &pool_info, NULL,
                                 &pool_handle);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&queue->vk);
      return result;
   }
   queue->cmd_pool = kk_cmd_pool_from_handle(pool_handle);

   queue->free_cmd_buffers = UTIL_DYNARRAY_INIT;
   queue->commits_in_flight = 0u;
   queue->merge_bookends = UTIL_DYNARRAY_INIT;
   queue->merge_commits_in_flight = 0u;
   mtx_init(&queue->mutex, mtx_plain);
   cnd_init(&queue->cond);

   queue->mtl_handle = mtl_new_command_queue(dev->mtl_handle);
   queue->merge_fence = mtl_new_fence(dev->mtl_handle);
   queue->vertex_wait = MTL_STAGE_ALL;
   mtl_command_queue_add_residency_set(queue->mtl_handle,
                                       dev->residency_set.handle);

   queue->vk.driver_submit = kk_queue_submit;

   return VK_SUCCESS;
}

void
kk_queue_finish(struct kk_device *dev, struct kk_queue *queue)
{
   /* There may be in-flight command buffers, wait for them to finish. */
   kk_queue_wait_for_commits(queue);

   mtl_command_queue_remove_residency_set(queue->mtl_handle,
                                          dev->residency_set.handle);
   mtl_release(queue->mtl_handle);
   mtl_release(queue->merge_fence);

   kk_DestroyCommandPool(kk_device_to_handle(dev),
                         kk_cmd_pool_to_handle(queue->cmd_pool), NULL);
   util_dynarray_foreach(&queue->merge_bookends, struct kk_bookend, bookend) {
      mtl_release(bookend->cmd_buf);
      mtl_release(bookend->allocator);
   }
   util_dynarray_fini(&queue->merge_bookends);
   util_dynarray_fini(&queue->free_cmd_buffers);
   cnd_destroy(&queue->cond);
   mtx_destroy(&queue->mutex);

   vk_queue_finish(&queue->vk);
}
