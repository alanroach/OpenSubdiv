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

#ifndef OSD_GLSL_COMPUTE_CONTROLLER_H
#define OSD_GLSL_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/dispatcher.h"
#include "../osd/glslComputeContext.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLSLComputeKernelBundle;

/// \brief Compute controller for launching GLSLCompute transform feedback
/// subdivision kernels.
///
/// OsdGLSLComputeController is a compute controller class to launch
/// GLSLCompute transfrom feedback subdivision kernels. It requires
/// OsdGLVertexBufferInterface as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OsdGLSLComputeController {
public:
    typedef OsdGLSLComputeContext ComputeContext;

    /// Constructor.
    OsdGLSLComputeController();

    /// Destructor.
    ~OsdGLSLComputeController();

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
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(OsdGLSLComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer) {

        if (batches.empty()) return;

        int numVertexElements = vertexBuffer ? vertexBuffer->GetNumElements() : 0;
        int numVaryingElements = varyingBuffer ? varyingBuffer->GetNumElements() : 0;

        bind(vertexBuffer, varyingBuffer, getKernels(numVertexElements, numVaryingElements));
        // bind table buffers.
        context->BindShaderStorageBuffers();

        FarDispatcher::Refine(this, context, batches, /*maxlevel*/-1);

        // unbind table buffers.
        context->UnbindShaderStorageBuffers();
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
    template<class VERTEX_BUFFER>
    void Refine(OsdGLSLComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer) {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)NULL);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

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

    OsdGLSLComputeKernelBundle * getKernels(int numVertexElements,
                                     int numVaryingElements);

    void bindBufferAndProgram();

    void unbindBufferAndProgram();

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying, OsdGLSLComputeKernelBundle *kernelBundle) {

        _currentVertexBuffer = vertex ? vertex->BindVBO() : 0;
        _currentVaryingBuffer = varying ? varying->BindVBO() : 0;

        _vdesc.numVertexElements = vertex ? vertex->GetNumElements() : 0;
        _vdesc.numVaryingElements = varying ? varying->GetNumElements() : 0;

        _currentKernelBundle = kernelBundle;

        bindBufferAndProgram();
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;
    }

private:
    std::vector<OsdGLSLComputeKernelBundle *> _kernelRegistry;

    GLuint _currentVertexBuffer, _currentVaryingBuffer;

    OsdVertexDescriptor _vdesc;

    OsdGLSLComputeKernelBundle * _currentKernelBundle;

};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_COMPUTE_CONTROLLER_H
