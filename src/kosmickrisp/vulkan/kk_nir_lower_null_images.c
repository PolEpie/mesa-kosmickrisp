/*
 * Copyright © 2025 Imagination Technologies Ltd.
 * Copyright (C) 2020-2021 Collabora, Ltd.
 * Copyright © 2020 Valve Corporation
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

static nir_def *
tex_handle_to_resource_id(nir_builder *b, nir_def *handle)
{
   /* Work backwards from the handle to the descriptor address it was loaded
    * from, and load the plain resource ID */
   const nir_instr *instr = nir_def_instr_const(handle);
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return nir_load_global_constant(b, 1, 64, intr->src[0].ssa);
}

static int
null_texture_slot(const nir_intrinsic_instr *handle)
{
   const bool array = nir_intrinsic_image_array(handle);
   enum mtl_texture_type type;
   switch (nir_intrinsic_image_dim(handle)) {
   case GLSL_SAMPLER_DIM_1D:
      type = array ? MTL_TEXTURE_TYPE_1D_ARRAY : MTL_TEXTURE_TYPE_1D;
      break;
   case GLSL_SAMPLER_DIM_2D:
      type = array ? MTL_TEXTURE_TYPE_2D_ARRAY : MTL_TEXTURE_TYPE_2D;
      break;
   case GLSL_SAMPLER_DIM_CUBE:
      type = array ? MTL_TEXTURE_TYPE_CUBE_ARRAY : MTL_TEXTURE_TYPE_CUBE;
      break;
   case GLSL_SAMPLER_DIM_3D:
      type = MTL_TEXTURE_TYPE_3D;
      break;
   default:
      return -1;
   }

   enum kk_null_texture_kind kind;
   if (handle->intrinsic == nir_intrinsic_load_depth_texture_kk) {
      if (type == MTL_TEXTURE_TYPE_1D || type == MTL_TEXTURE_TYPE_1D_ARRAY ||
          type == MTL_TEXTURE_TYPE_3D)
         return -1;
      kind = KK_NULL_TEXTURE_DEPTH;
   } else {
      switch (nir_alu_type_get_base_type(nir_intrinsic_dest_type(handle))) {
      case nir_type_float:
         kind = KK_NULL_TEXTURE_FLOAT;
         break;
      case nir_type_uint:
         kind = KK_NULL_TEXTURE_UINT;
         break;
      case nir_type_int:
         kind = KK_NULL_TEXTURE_SINT;
         break;
      default:
         return -1;
      }
   }

   return type * KK_NULL_TEXTURE_KINDS + kind;
}

static bool
lower_select(nir_builder *b, nir_instr *instr, nir_src *handle_src,
             nir_def *def)
{
   nir_instr *handle_instr = nir_def_instr(handle_src->ssa);
   if (handle_instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *handle = nir_instr_as_intrinsic(handle_instr);
   if (handle->intrinsic != nir_intrinsic_load_texture_handle_kk &&
       handle->intrinsic != nir_intrinsic_load_depth_texture_kk)
      return false;

   const int slot = null_texture_slot(handle);
   if (slot < 0)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *addr = handle->src[0].ssa;
   nir_def *is_null =
      nir_ieq_imm(b, tex_handle_to_resource_id(b, &handle->def), 0);
   nir_def *root = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   nir_def *table = nir_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, root,
                   offsetof(struct kk_root_descriptor_table, null_textures)));
   nir_def *null_addr = nir_iadd_imm(b, table, slot * sizeof(uint64_t));
   nir_def *safe_addr = nir_bcsel(b, is_null, null_addr, addr);

   nir_intrinsic_instr *safe =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, handle_instr));
   nir_builder_instr_insert(b, &safe->instr);
   nir_src_rewrite(&safe->src[0], safe_addr);
   nir_src_rewrite(handle_src, &safe->def);

   if (def) {
      b->cursor = nir_after_instr(instr);
      nir_def *zero = nir_imm_zero(b, def->num_components, def->bit_size);
      nir_def_rewrite_uses_after(def, nir_bcsel(b, is_null, zero, def));
   }

   return true;
}

static nir_def *
get_is_null(nir_builder *b, nir_instr *instr, nir_def **def)
{
   *def = NULL;
   nir_def *handle = NULL;

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_bindless_image_load:
      case nir_intrinsic_bindless_image_sparse_load:
      case nir_intrinsic_bindless_image_store:
      case nir_intrinsic_bindless_image_atomic:
      case nir_intrinsic_bindless_image_atomic_swap:
      case nir_intrinsic_bindless_image_size:
      case nir_intrinsic_bindless_image_samples:
      case nir_intrinsic_bindless_image_levels:
         if (nir_intrinsic_infos[intr->intrinsic].has_dest)
            *def = &intr->def;

         handle = intr->src[0].ssa;
         break;

      default:
         break;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      *def = &tex->def;

      handle = nir_get_tex_src(tex, nir_tex_src_texture_handle);
   }

   if (!handle)
      return NULL;

   nir_def *resource_id = tex_handle_to_resource_id(b, handle);
   return nir_ieq_imm(b, resource_id, 0);
}

static bool
lower(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      const int index =
         nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
      if (index >= 0 &&
          lower_select(b, instr, &tex->src[index].src, &tex->def))
         return true;
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_bindless_image_load:
      case nir_intrinsic_bindless_image_sparse_load:
      case nir_intrinsic_bindless_image_size:
      case nir_intrinsic_bindless_image_samples:
      case nir_intrinsic_bindless_image_levels:
         if (lower_select(b, instr, &intr->src[0], &intr->def))
            return true;
         break;
      default:
         break;
      }
   }

   b->cursor = nir_before_instr(instr);

   nir_def *def;
   nir_def *is_null = get_is_null(b, instr, &def);

   if (!is_null)
      return false;

   nir_def *zero = NULL;
   nir_if *nif = nir_push_if(b, nir_inot(b, is_null));
   nir_instr_remove(instr);
   nir_builder_instr_insert(b, instr);
   if (def) {
      nir_push_else(b, nif);
      zero = nir_imm_zero(b, def->num_components, def->bit_size);
   }
   nir_pop_if(b, nif);

   if (def) {
      nir_def *phi = nir_if_phi(b, def, zero);

      /* We can't use nir_def_rewrite_uses_after on phis, so use the global
       * version and fixup the phi manually
       */
      nir_def_rewrite_uses(def, phi);

      nir_instr *phi_instr = nir_def_instr(phi);
      nir_phi_instr *phi_as_phi = nir_instr_as_phi(phi_instr);
      nir_phi_src *phi_src =
         nir_phi_get_src_from_block(phi_as_phi, instr->block);
      nir_src_rewrite(&phi_src->src, def);
   }

   return true;
}

bool
kk_nir_lower_null_images(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower, nir_metadata_none, NULL);
}
