//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#ifndef OSD_CL_COMPUTE_CONTROLLER_H
#define OSD_CL_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/dispatcher.h"
#include "../osd/clComputeContext.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/opencl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCLKernelBundle;

/// \brief Compute controller for launching OpenCL subdivision kernels.
///
/// OsdCLComputeController is a compute controller class to launch
/// OpenCL subdivision kernels. It requires OsdCLVertexBufferInterface
/// as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OsdCLComputeController {
public:
    typedef OsdCLComputeContext ComputeContext;

    /// Constructor.
    ///
    /// @param clContext a valid instanciated OpenCL context
    ///
    /// @param queue a valid non-zero OpenCL command queue
    ///
    OsdCLComputeController(cl_context clContext, cl_command_queue queue);

    /// Destructor.
    ~OsdCLComputeController();

    /// Launch subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       the OsdCpuContext to apply refinement operations to
    ///
    /// @param  batches       vector of batches of vertices organized by operative 
    ///                       kernel
    ///
    /// @param  vertexBuffer  vertex-interpolated data buffer
    ///
    /// @param  varyingBuffer varying-interpolated data buffer
    ///
    /// @param  vertexDesc    the descriptor of vertex elements to be refined.
    ///                       if it's null, all primvars in the vertex buffer
    ///                       will be refined.
    ///
    /// @param  varyingDesc   the descriptor of varying elements to be refined.
    ///                       if it's null, all primvars in the varying buffer
    ///                       will be refined.
    ///
    /// @param numStartEvents the number of events in the array pointed to by
    ///                       startEvents.
    ///
    /// @param startEvents    points to an array of cl_event which will determine
    ///                       when it is safe for the OpenCL device to begin work
	///                       or NULL if it can begin immediately.
    ///
    /// @param finalEvent     pointer to a cl_event which will recieve a copy of
    ///                       the cl_event which indicates when all work for this
    ///                       call has completed.  This cl_event has an incremented
    ///                       reference count and should be released via
    ///                       clReleaseEvent().  NULL if not required.
    ///
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(ComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer,
                OsdVertexBufferDescriptor const *vertexDesc=NULL,
                OsdVertexBufferDescriptor const *varyingDesc=NULL,
                unsigned int numStartEvents=0,
                const cl_event* startEvents=NULL,
                cl_event* finalEvent=NULL) {

        if (batches.empty()) return;

        bind(vertexBuffer, varyingBuffer, vertexDesc, varyingDesc, numStartEvents, startEvents, finalEvent, _useSyncEvents);

        FarDispatcher::Refine(this, context, batches, /*maxlevel*/-1);

        unbind();
    }

    /// Launch subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       the OsdCpuContext to apply refinement operations to
    ///
    /// @param  batches       vector of batches of vertices organized by operative 
    ///                       kernel
    ///
    /// @param  vertexBuffer  vertex-interpolated data buffer
    ///
    /// @param numStartEvents the number of events in the array pointed to by
    ///                       startEvents.
    ///
    /// @param startEvents    points to an array of cl_event which will determine
    ///                       when it is safe for the OpenCL device to begin work
	///                       or NULL if it can begin immediately.
    ///
    /// @param finalEvent     pointer to a cl_event which will recieve a copy of
    ///                       the cl_event which indicates when all work for this
    ///                       call has completed.  This cl_event has an incremented
    ///                       reference count and should be released via
    ///                       clReleaseEvent().  NULL if not required.
    ///
    template<class VERTEX_BUFFER>
    void Refine(ComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                unsigned int numStartEvents=0,
                const cl_event* startEvent=NULL,
                cl_event* finalEvent=NULL) {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)NULL, NULL, NULL, numStartEvents, startEvent, finalEvent);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

    /// Returns CL context
    cl_context GetContext() const { return _clContext; }

    /// Returns CL command queue
    cl_command_queue GetCommandQueue() const { return _clQueue; }

protected:
    friend class FarDispatcher;

    void ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyVertexEdits(FarKernelBatch const &batch, ComputeContext const *context) const;


    OsdCLKernelBundle * getKernelBundle(
        OsdVertexBufferDescriptor const &vertexDesc,
        OsdVertexBufferDescriptor const &varyingDesc);

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying,
              OsdVertexBufferDescriptor const *vertexDesc,
              OsdVertexBufferDescriptor const *varyingDesc,
              unsigned int numStartEvents, const cl_event* startEvents, cl_event* finalEvent, bool forceUseEvents) {

        // if the vertex buffer descriptor is specified, use it.
        // otherwise, assumes the data is tightly packed in the vertex buffer.
        if (vertexDesc) {
            _currentBindState.vertexDesc = *vertexDesc;
        } else {
            int numElements = vertex ? vertex->GetNumElements() : 0;
            _currentBindState.vertexDesc = OsdVertexBufferDescriptor(
                0, numElements, numElements);
        }
        if (varyingDesc) {
            _currentBindState.varyingDesc = *varyingDesc;
        } else {
            int numElements = varying ? varying->GetNumElements() : 0;
            _currentBindState.varyingDesc = OsdVertexBufferDescriptor(
                0, numElements, numElements);
        }

        _currentBindState.vertexBuffer = vertex ? vertex->BindCLBuffer(_clQueue) : 0;
        _currentBindState.varyingBuffer = varying ? varying->BindCLBuffer(_clQueue) : 0;
        _currentBindState.kernelBundle = getKernelBundle(_currentBindState.vertexDesc,
                                                         _currentBindState.varyingDesc);

        bool useEvents = forceUseEvents || numStartEvents > 0 || finalEvent != NULL;
        if (useEvents) {
            _currentBindState.numStartEvents = numStartEvents;
            _currentBindState.startEvents = (numStartEvents > 0) ? startEvents : NULL;
            _currentBindState.endEvent = &_currentBindState.outEvent;
            _currentBindState.finalEvent = finalEvent;
        }
        // If not using events, then event pointers passed to CL functions (startEvents
        // and endEvent) remain 0.
    }

    void unbind() {
        if (_currentBindState.inEvent) {
            _completionEvents.push_back(_currentBindState.inEvent);

            if (_currentBindState.finalEvent) {
                // Pass the last intermediate event in the chain back to the caller and
                // increment the ref count since we keep a copy.
                *_currentBindState.finalEvent = _currentBindState.inEvent;
                clRetainEvent(_currentBindState.inEvent);
            }
        }

        _currentBindState.Reset();
    }

private:
	void postEnqueueKernel() const;

    struct BindState {
        BindState() : vertexBuffer(NULL), varyingBuffer(NULL), kernelBundle(NULL),
						numStartEvents(0), startEvents(NULL), finalEvent(NULL), inEvent(NULL), outEvent(NULL), endEvent(NULL) {}
        void Reset() {
            vertexBuffer = varyingBuffer = NULL;
            numStartEvents = 0;
            startEvents = NULL;
            finalEvent = NULL;
            inEvent = NULL;
            outEvent = NULL;
            endEvent = NULL;
            vertexDesc.Reset();
            varyingDesc.Reset();
            kernelBundle = NULL;
        }

        cl_mem vertexBuffer;
        cl_mem varyingBuffer;
		
        unsigned int numStartEvents;
        const cl_event* startEvents;
        cl_event* finalEvent;

        cl_event inEvent;
        cl_event outEvent;
        cl_event* endEvent;

        OsdVertexBufferDescriptor vertexDesc;
        OsdVertexBufferDescriptor varyingDesc;
        OsdCLKernelBundle *kernelBundle;
    };

    BindState _currentBindState;

    cl_context _clContext;
    cl_command_queue _clQueue;
    std::vector<OsdCLKernelBundle *> _kernelRegistry;

    bool _useSyncEvents;
    std::vector<cl_event> _completionEvents;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_COMPUTE_CONTROLLER_H
