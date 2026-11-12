/*
 * Copyright (C) 2014-2015 Michael Müller
 * Copyright (C) 2023-2024 Sveinar Søpler
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#ifndef __WINE_NVCUDA_H
#define __WINE_NVCUDA_H

#include "windef.h"
#include "winbase.h"
#include "cuda.h"

void __attribute((visibility("hidden"))) cuda_process_tls_callbacks(DWORD reason);
CUresult __attribute((visibility("hidden"))) cuda_get_table(const void **table, const CUuuid *uuid, const void *orig_table, CUresult orig_result);
int __attribute((visibility("hidden"))) get_shared_resource_fd(HANDLE win32_handle);
int __attribute((visibility("hidden"))) get_d3dkmt_object_fd(HANDLE win32_handle, BOOL sync);

/* nvcuda_d3d11.c: D3D11 buffers registered with the CUDA graphics API. */
struct ID3D11Resource;
CUresult __attribute((visibility("hidden"))) d3d11_register_resource(CUgraphicsResource *handle, struct ID3D11Resource *d3d_resource, unsigned int flags);
BOOL __attribute((visibility("hidden"))) d3d11_is_resource(CUgraphicsResource handle);
CUresult __attribute((visibility("hidden"))) d3d11_unregister_resource(CUgraphicsResource handle);
CUresult __attribute((visibility("hidden"))) d3d11_map_resource(CUgraphicsResource handle, CUstream stream, BOOL ptsz);
CUresult __attribute((visibility("hidden"))) d3d11_unmap_resource(CUgraphicsResource handle, CUstream stream, BOOL ptsz, BOOL copy_back);
CUresult __attribute((visibility("hidden"))) d3d11_get_mapped_pointer(CUdeviceptr *pointer, size_t *size, CUgraphicsResource handle);
CUresult __attribute((visibility("hidden"))) d3d11_set_map_flags(CUgraphicsResource handle, unsigned int flags);
CUresult __attribute((visibility("hidden"))) d3d11_not_an_array(CUgraphicsResource handle);
HANDLE __attribute((visibility("hidden"))) get_shared_resource_kmt_handle(HANDLE shared_resource);

#endif
