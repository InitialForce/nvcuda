/*
 * D3D11 buffer interop for the CUDA graphics API (cuGraphicsD3D11RegisterResource)
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

/*
 * Linux libcuda has no Direct3D interop, and under Wine a D3D11 resource is a DXVK Vulkan
 * resource. A registered D3D11 buffer is therefore shared with CUDA through a twin:
 *
 *   register   a twin buffer of the same size, created on the same D3D11 device with
 *              D3D11_RESOURCE_MISC_SHARED_NTHANDLE. DXVK gives it exportable memory, Wine
 *              turns its NT handle into an fd, and CUDA imports that fd as external memory.
 *              A shared D3D11 fence, imported as a CUDA timeline semaphore, orders the two.
 *   map        D3D11 copies the buffer into the twin and signals the fence; the CUDA stream
 *              waits for that fence value. CUDA then sees the buffer contents at the time of
 *              the map, as the CUDA graphics API requires.
 *   unmap      the CUDA stream signals the fence; D3D11 waits for it and copies the twin back
 *              into the buffer, unless the resource is read-only for CUDA.
 *
 * Both directions stay on the GPU: the CPU never waits for either queue. The cost is one
 * buffer copy per map, and one per unmap when CUDA may write. Textures are not supported.
 */

#include <stdarg.h>
#include <stdint.h>
#include <limits.h>
#include <dlfcn.h>
#include <unistd.h>

#define COBJMACROS
#include "windef.h"
#include "winbase.h"
#include "winternl.h"
#include "wine/debug.h"
#include "wine/list.h"
#include "initguid.h"
#include "d3d11_4.h"
#include "dxgi1_2.h"
#include "cuda.h"
#include "nvcuda.h"

WINE_DEFAULT_DEBUG_CHANNEL(nvcuda);

#define CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY     0x1
#define CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD 0x2
#define CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          0x0
#define CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     0x1
#define CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD 0x2

#define CUDA_SUCCESS_                0
#define CUDA_ERROR_OUT_OF_MEMORY_    2
#define CUDA_ERROR_UNKNOWN_          999
#define CUDA_ERROR_ALREADY_MAPPED_   208
#define CUDA_ERROR_NOT_MAPPED_       211
#define CUDA_ERROR_NOT_MAPPED_AS_ARRAY_ 213
#define CUDA_ERROR_NOT_SUPPORTED_    801
#define CUDA_EXTERNAL_MEMORY_DEDICATED_ 0x1

typedef struct CUextMemory_st *CUexternalMemory_;

typedef struct
{
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC_;

typedef struct
{
    struct
    {
        struct { unsigned long long value; } fence;
        union { void *fence; unsigned long long reserved; } nvSciSync;
        struct { unsigned long long key; unsigned int timeoutMs; } keyedMutex;
        unsigned int reserved[10];
    } params;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_PARAMS_;

struct d3d11_cuda_resource
{
    struct list entry;
    ID3D11Buffer *buffer;
    ID3D11Buffer *twin;
    ID3D11DeviceContext4 *context;
    ID3D11Fence *fence;
    ID3D11Multithread *multithread;
    UINT64 fence_value;
    CUexternalMemory_ memory;
    CUexternalSemaphore semaphore;
    CUdeviceptr pointer;
    size_t size;
    unsigned int register_flags;
    unsigned int map_flags;
    BOOL mapped;
};

static struct list resources = LIST_INIT(resources);
static CRITICAL_SECTION resources_cs;
static CRITICAL_SECTION_DEBUG resources_cs_debug =
{
    0, 0, &resources_cs,
    { &resources_cs_debug.ProcessLocksList, &resources_cs_debug.ProcessLocksList },
      0, 0, { (DWORD_PTR)(__FILE__ ": resources_cs") }
};
static CRITICAL_SECTION resources_cs = { &resources_cs_debug, -1, 0, 0, 0, 0 };

static CUresult (*p_cuImportExternalMemory)(CUexternalMemory_ *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
static CUresult (*p_cuExternalMemoryGetMappedBuffer)(CUdeviceptr *, CUexternalMemory_, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_ *);
static CUresult (*p_cuDestroyExternalMemory)(CUexternalMemory_);
static CUresult (*p_cuImportExternalSemaphore)(CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
static CUresult (*p_cuDestroyExternalSemaphore)(CUexternalSemaphore);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuSignal[2])(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_PARAMS_ *, unsigned int, CUstream);
static CUresult (*p_cuWait[2])(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_PARAMS_ *, unsigned int, CUstream);

static BOOL driver_functions_loaded;

static BOOL WINAPI load_driver_functions_once(INIT_ONCE *once, void *param, void **context)
{
    void *libcuda;

    if (!(libcuda = dlopen("libcuda.so.1", RTLD_NOW)))
    {
        ERR("Can't open libcuda.so.1: %s\n", dlerror());
        return TRUE;
    }
#define LOAD(var, name) if (!(*(void **)&(var) = dlsym(libcuda, name))) { ERR("Missing %s in libcuda.\n", name); return TRUE; }
    LOAD(p_cuImportExternalMemory, "cuImportExternalMemory");
    LOAD(p_cuExternalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer");
    LOAD(p_cuDestroyExternalMemory, "cuDestroyExternalMemory");
    LOAD(p_cuImportExternalSemaphore, "cuImportExternalSemaphore");
    LOAD(p_cuDestroyExternalSemaphore, "cuDestroyExternalSemaphore");
    LOAD(p_cuMemFree, "cuMemFree_v2");
    LOAD(p_cuSignal[0], "cuSignalExternalSemaphoresAsync");
    LOAD(p_cuSignal[1], "cuSignalExternalSemaphoresAsync_ptsz");
    LOAD(p_cuWait[0], "cuWaitExternalSemaphoresAsync");
    LOAD(p_cuWait[1], "cuWaitExternalSemaphoresAsync_ptsz");
#undef LOAD
    driver_functions_loaded = TRUE;
    return TRUE;
}

/* Several threads register resources at once, so the libcuda lookup runs exactly once. */
static BOOL load_driver_functions(void)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;

    InitOnceExecuteOnce(&once, load_driver_functions_once, NULL, NULL);
    return driver_functions_loaded;
}

/* Returns our resource for a CUgraphicsResource, or NULL when the driver owns it. */
static struct d3d11_cuda_resource *find_resource(CUgraphicsResource handle)
{
    struct d3d11_cuda_resource *resource;

    LIST_FOR_EACH_ENTRY(resource, &resources, struct d3d11_cuda_resource, entry)
        if ((CUgraphicsResource)resource == handle) return resource;
    return NULL;
}

static void destroy_resource(struct d3d11_cuda_resource *resource)
{
    if (resource->pointer) p_cuMemFree(resource->pointer);
    if (resource->memory) p_cuDestroyExternalMemory(resource->memory);
    if (resource->semaphore) p_cuDestroyExternalSemaphore(resource->semaphore);
    if (resource->fence) ID3D11Fence_Release(resource->fence);
    if (resource->context) ID3D11DeviceContext4_Release(resource->context);
    if (resource->multithread) ID3D11Multithread_Release(resource->multithread);
    if (resource->twin) ID3D11Buffer_Release(resource->twin);
    if (resource->buffer) ID3D11Buffer_Release(resource->buffer);
    HeapFree(GetProcessHeap(), 0, resource);
}

/* CUDA imports a whole Vulkan allocation, so it needs the allocation's size, which D3D11 does not
 * report. The twin is a whole number of 64 KiB pages, a size that needs no padding, so its size
 * is the allocation size. Only the first ByteWidth bytes are copied and mapped. */
#define TWIN_ALIGN 0x10000

static CUresult import_twin_memory(struct d3d11_cuda_resource *resource, ID3D11Device *device,
                                   const D3D11_BUFFER_DESC *desc)
{
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC_ buffer_desc = {0};
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory_desc = {0};
    D3D11_BUFFER_DESC twin_desc = {0};
    IDXGIResource1 *dxgi = NULL;
    HANDLE shared = NULL;
    CUresult ret;
    HRESULT hr;
    int fd;

    if (desc->ByteWidth > UINT_MAX - (TWIN_ALIGN - 1)) return CUDA_ERROR_INVALID_VALUE;
    twin_desc.ByteWidth = (desc->ByteWidth + TWIN_ALIGN - 1) & ~(TWIN_ALIGN - 1);
    twin_desc.Usage = D3D11_USAGE_DEFAULT;
    twin_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

    if (FAILED(hr = ID3D11Device_CreateBuffer(device, &twin_desc, NULL, &resource->twin)))
    {
        ERR("Can't create a shared %u-byte twin buffer, hr %#lx.\n", desc->ByteWidth, (long)hr);
        return CUDA_ERROR_OUT_OF_MEMORY_;
    }
    if (FAILED(hr = ID3D11Buffer_QueryInterface(resource->twin, &IID_IDXGIResource1, (void **)&dxgi))
        || FAILED(hr = IDXGIResource1_CreateSharedHandle(dxgi, NULL, GENERIC_ALL, NULL, &shared)))
    {
        ERR("Can't share the twin buffer, hr %#lx. This needs DXVK with shared buffer support.\n", (long)hr);
        if (dxgi) IDXGIResource1_Release(dxgi);
        return CUDA_ERROR_NOT_SUPPORTED_;
    }
    IDXGIResource1_Release(dxgi);

    fd = get_d3dkmt_object_fd(shared, FALSE);
    CloseHandle(shared);
    if (fd < 0) return CUDA_ERROR_UNKNOWN_;

    memory_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memory_desc.handle.fd = fd;
    memory_desc.size = twin_desc.ByteWidth;
    memory_desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED_;

    /* On success CUDA owns the fd. */
    if ((ret = p_cuImportExternalMemory(&resource->memory, &memory_desc)))
    {
        ERR("CUDA can't import the twin buffer memory (%u bytes), error %d.\n", twin_desc.ByteWidth, ret);
        close(fd);
        resource->memory = NULL;
        return ret;
    }

    buffer_desc.offset = 0;
    buffer_desc.size = desc->ByteWidth;
    if ((ret = p_cuExternalMemoryGetMappedBuffer(&resource->pointer, resource->memory, &buffer_desc)))
    {
        ERR("CUDA can't map the twin buffer memory, error %d.\n", ret);
        resource->pointer = 0;
        return ret;
    }

    resource->size = desc->ByteWidth;
    return CUDA_SUCCESS_;
}

static CUresult import_fence(struct d3d11_cuda_resource *resource, ID3D11Device *device)
{
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semaphore_desc = {0};
    ID3D11Device5 *device5 = NULL;
    HANDLE shared = NULL;
    CUresult ret;
    HRESULT hr;
    int fd;

    if (FAILED(hr = ID3D11Device_QueryInterface(device, &IID_ID3D11Device5, (void **)&device5)))
    {
        ERR("The D3D11 device has no ID3D11Device5, hr %#lx.\n", (long)hr);
        return CUDA_ERROR_NOT_SUPPORTED_;
    }
    hr = ID3D11Device5_CreateFence(device5, 0, D3D11_FENCE_FLAG_SHARED, &IID_ID3D11Fence, (void **)&resource->fence);
    ID3D11Device5_Release(device5);
    if (FAILED(hr) || FAILED(hr = ID3D11Fence_CreateSharedHandle(resource->fence, NULL, GENERIC_ALL, NULL, &shared)))
    {
        ERR("Can't create a shared D3D11 fence, hr %#lx.\n", (long)hr);
        return CUDA_ERROR_NOT_SUPPORTED_;
    }

    fd = get_d3dkmt_object_fd(shared, TRUE);
    CloseHandle(shared);
    if (fd < 0) return CUDA_ERROR_UNKNOWN_;

    semaphore_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
    semaphore_desc.handle.fd = fd;
    if ((ret = p_cuImportExternalSemaphore(&resource->semaphore, &semaphore_desc)))
    {
        ERR("CUDA can't import the D3D11 fence as a timeline semaphore, error %d.\n", ret);
        close(fd);
        resource->semaphore = NULL;
        return ret;
    }
    resource->fence_value = 0;
    return CUDA_SUCCESS_;
}

CUresult d3d11_register_resource(CUgraphicsResource *handle, ID3D11Resource *d3d_resource, unsigned int flags)
{
    struct d3d11_cuda_resource *resource;
    D3D11_RESOURCE_DIMENSION dimension;
    ID3D11DeviceContext *context;
    ID3D11Device *device = NULL;
    D3D11_BUFFER_DESC desc;
    CUresult ret;

    TRACE("(%p, %p, %#x)\n", handle, d3d_resource, flags);

    if (!handle || !d3d_resource) return CUDA_ERROR_INVALID_VALUE;
    if (!load_driver_functions()) return CUDA_ERROR_UNKNOWN_;

    ID3D11Resource_GetType(d3d_resource, &dimension);
    if (dimension != D3D11_RESOURCE_DIMENSION_BUFFER)
    {
        FIXME("Only D3D11 buffers can be registered, not resource dimension %u.\n", dimension);
        return CUDA_ERROR_NOT_SUPPORTED_;
    }
    if (flags & ~(CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY | CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))
    {
        FIXME("Unsupported register flags %#x.\n", flags);
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (!(resource = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*resource))))
        return CUDA_ERROR_OUT_OF_MEMORY_;
    resource->register_flags = flags;
    resource->map_flags = flags;

    ID3D11Resource_QueryInterface(d3d_resource, &IID_ID3D11Buffer, (void **)&resource->buffer);
    ID3D11Buffer_GetDesc(resource->buffer, &desc);
    ID3D11Buffer_GetDevice(resource->buffer, &device);
    ID3D11Device_GetImmediateContext(device, &context);
    ID3D11DeviceContext_QueryInterface(context, &IID_ID3D11DeviceContext4, (void **)&resource->context);
    ID3D11DeviceContext_QueryInterface(context, &IID_ID3D11Multithread, (void **)&resource->multithread);
    ID3D11DeviceContext_Release(context);

    if (!resource->context) ret = CUDA_ERROR_NOT_SUPPORTED_;
    else if (!(ret = import_twin_memory(resource, device, &desc))) ret = import_fence(resource, device);
    ID3D11Device_Release(device);

    if (ret)
    {
        destroy_resource(resource);
        return ret;
    }

    EnterCriticalSection(&resources_cs);
    list_add_tail(&resources, &resource->entry);
    LeaveCriticalSection(&resources_cs);

    TRACE("Registered D3D11 buffer %p (%u bytes) as %p, CUDA pointer %#llx.\n",
          d3d_resource, desc.ByteWidth, resource, (unsigned long long)resource->pointer);
    *handle = (CUgraphicsResource)resource;
    return CUDA_SUCCESS_;
}

BOOL d3d11_is_resource(CUgraphicsResource handle)
{
    BOOL ret;

    EnterCriticalSection(&resources_cs);
    ret = find_resource(handle) != NULL;
    LeaveCriticalSection(&resources_cs);
    return ret;
}

CUresult d3d11_unregister_resource(CUgraphicsResource handle)
{
    struct d3d11_cuda_resource *resource;

    EnterCriticalSection(&resources_cs);
    if ((resource = find_resource(handle))) list_remove(&resource->entry);
    LeaveCriticalSection(&resources_cs);

    if (!resource) return CUDA_ERROR_INVALID_HANDLE;
    destroy_resource(resource);
    return CUDA_SUCCESS_;
}

/* The immediate context belongs to the application. Its multithread lock (when the application
 * turned protection on) keeps our calls from running at the same time as the application's. */
static void lock_context(struct d3d11_cuda_resource *resource)
{
    if (resource->multithread) ID3D11Multithread_Enter(resource->multithread);
}

static void unlock_context(struct d3d11_cuda_resource *resource)
{
    if (resource->multithread) ID3D11Multithread_Leave(resource->multithread);
}

/* Copies the application's ByteWidth bytes between the buffer and the larger twin. */
static void copy_buffer(struct d3d11_cuda_resource *resource, ID3D11Buffer *dst, ID3D11Buffer *src)
{
    D3D11_BOX box = { 0, 0, 0, resource->size, 1, 1 };

    ID3D11DeviceContext4_CopySubresourceRegion(resource->context, (ID3D11Resource *)dst, 0, 0, 0, 0,
                                               (ID3D11Resource *)src, 0, &box);
}

CUresult d3d11_map_resource(CUgraphicsResource handle, CUstream stream, BOOL ptsz)
{
    CUDA_EXTERNAL_SEMAPHORE_PARAMS_ params = {0};
    struct d3d11_cuda_resource *resource;
    CUresult ret;
    HRESULT hr;

    EnterCriticalSection(&resources_cs);
    if (!(resource = find_resource(handle)))
    {
        LeaveCriticalSection(&resources_cs);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (resource->mapped)
    {
        LeaveCriticalSection(&resources_cs);
        return CUDA_ERROR_ALREADY_MAPPED_;
    }

    lock_context(resource);
    if (!(resource->map_flags & CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD))
        copy_buffer(resource, resource->twin, resource->buffer);
    params.params.fence.value = resource->fence_value + 1;
    hr = ID3D11DeviceContext4_Signal(resource->context, resource->fence, params.params.fence.value);
    ID3D11DeviceContext4_Flush(resource->context);
    unlock_context(resource);

    /* Without the signal, the CUDA wait below would never end. */
    if (FAILED(hr))
    {
        ERR("Can't signal the D3D11 fence, hr %#lx.\n", (long)hr);
        LeaveCriticalSection(&resources_cs);
        return CUDA_ERROR_UNKNOWN_;
    }
    resource->fence_value++;

    if ((ret = p_cuWait[ptsz](&resource->semaphore, &params, 1, stream)))
        ERR("CUDA can't wait for the D3D11 fence, error %d.\n", ret);
    else
        resource->mapped = TRUE;

    LeaveCriticalSection(&resources_cs);
    return ret;
}

/* With copy_back FALSE the D3D11 buffer keeps its contents: the rollback of a failed map uses
 * that, because after a WRITE_DISCARD map the twin holds stale bytes. */
CUresult d3d11_unmap_resource(CUgraphicsResource handle, CUstream stream, BOOL ptsz, BOOL copy_back)
{
    CUDA_EXTERNAL_SEMAPHORE_PARAMS_ params = {0};
    struct d3d11_cuda_resource *resource;
    CUresult ret;
    HRESULT hr;

    EnterCriticalSection(&resources_cs);
    if (!(resource = find_resource(handle)))
    {
        LeaveCriticalSection(&resources_cs);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!resource->mapped)
    {
        LeaveCriticalSection(&resources_cs);
        return CUDA_ERROR_NOT_MAPPED_;
    }

    params.params.fence.value = ++resource->fence_value;
    if ((ret = p_cuSignal[ptsz](&resource->semaphore, &params, 1, stream)))
    {
        ERR("CUDA can't signal the D3D11 fence, error %d.\n", ret);
        resource->fence_value--;
        LeaveCriticalSection(&resources_cs);
        return ret;
    }
    resource->mapped = FALSE;

    /* D3D11 work after this point waits on the GPU until CUDA is done with the twin. If the
     * GPU wait can't be queued, wait on the CPU instead, so the copy back never runs early. */
    lock_context(resource);
    if (FAILED(hr = ID3D11DeviceContext4_Wait(resource->context, resource->fence, params.params.fence.value)))
    {
        HANDLE event = CreateEventW(NULL, FALSE, FALSE, NULL);

        ERR("Can't make D3D11 wait for CUDA, hr %#lx; waiting on the CPU.\n", (long)hr);
        if (!event || FAILED(hr = ID3D11Fence_SetEventOnCompletion(resource->fence, params.params.fence.value, event))
            || WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0)
        {
            ERR("Can't wait for CUDA, hr %#lx; the buffer keeps its old contents.\n", (long)hr);
            ret = CUDA_ERROR_UNKNOWN_;
        }
        if (event) CloseHandle(event);
    }
    if (!ret && copy_back && !(resource->map_flags & CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY))
        copy_buffer(resource, resource->buffer, resource->twin);
    ID3D11DeviceContext4_Flush(resource->context);
    unlock_context(resource);

    LeaveCriticalSection(&resources_cs);
    return ret;
}

CUresult d3d11_get_mapped_pointer(CUdeviceptr *pointer, size_t *size, CUgraphicsResource handle)
{
    struct d3d11_cuda_resource *resource;
    CUresult ret = CUDA_SUCCESS_;

    EnterCriticalSection(&resources_cs);
    if (!(resource = find_resource(handle))) ret = CUDA_ERROR_INVALID_HANDLE;
    else if (!resource->mapped) ret = CUDA_ERROR_NOT_MAPPED_;
    else
    {
        if (pointer) *pointer = resource->pointer;
        if (size) *size = resource->size;
    }
    LeaveCriticalSection(&resources_cs);
    return ret;
}

CUresult d3d11_set_map_flags(CUgraphicsResource handle, unsigned int flags)
{
    struct d3d11_cuda_resource *resource;
    CUresult ret = CUDA_SUCCESS_;

    EnterCriticalSection(&resources_cs);
    if (!(resource = find_resource(handle))) ret = CUDA_ERROR_INVALID_HANDLE;
    else if (resource->mapped) ret = CUDA_ERROR_ALREADY_MAPPED_;
    else if (flags != CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE && flags != CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
             && flags != CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
        ret = CUDA_ERROR_INVALID_VALUE;
    else resource->map_flags = flags;
    LeaveCriticalSection(&resources_cs);
    return ret;
}

CUresult d3d11_not_an_array(CUgraphicsResource handle)
{
    return d3d11_is_resource(handle) ? CUDA_ERROR_NOT_MAPPED_AS_ARRAY_ : CUDA_ERROR_INVALID_HANDLE;
}
