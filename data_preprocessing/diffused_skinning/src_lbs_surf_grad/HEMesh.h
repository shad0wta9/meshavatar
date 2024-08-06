#ifndef __HEMESH_H__
#define __HEMESH_H__

#include <vector>
#include "DTypes.h"
#include "HEMath.h"
#include <cstdio>
#include <cassert>

#define N_SMPL_JOINTS 55
#define N_SMPL_VERTS 6890

// #define N_SMPL_JOINTS 55
// #define N_SMPL_VERTS 10475

namespace HEMesh {
    class MeshStruct;
    struct HEdge;
    struct Face;
    struct Vertex;
    bool loadFromOBJ(const char * obj_fn, std::vector<HEfloat> & out_verts, std::vector<HEint> & out_v_indices);
    bool loadWeights(const char * txt_fn, std::vector<HEfloat> & out_weights);
    bool writeToOBJ(const char * obj_fn, const std::vector<HEfloat> & out_verts, const std::vector<HEint> & out_faces);
    bool writeLBSWeightsToTxt(const char * obj_fn);
}

struct HEMesh::HEdge {
    HEint pair;
    HEint next;
    HEint f;
    HEint v;

    HEdge(HEint PAIR, HEint NEXT, HEint F, HEint V): pair(PAIR), next(NEXT), f(F), v(V) {}
};

struct HEMesh::Face {
    HEint h;
    HEMath::Vec3 n;
    // other data
    Face(HEint H): h(H), n(0.0f, 0.0f, 0.0f) {}
};

struct HEMesh::Vertex {
    HEint h;    // any hedge that points to this vertex
    HEMath::Vec3 p;
    HEMath::Vec3 n;   // normal vector
    HEMath::Vec3 tx;  // tangent vector
    HEMath::Vec3 ty;  // tangent vector
    HEfloat lbs_weights[N_SMPL_JOINTS] = {};
    HEfloat lbs_weights_grad_tx[N_SMPL_JOINTS] = {};
    HEfloat lbs_weights_grad_ty[N_SMPL_JOINTS] = {};
    // other data
    Vertex(HEint H, HEfloat X, HEfloat Y, HEfloat Z): h(H), p(X, Y, Z), n(0.0f, 0.0f, 0.0f) {}
    Vertex(HEint H, HEfloat X, HEfloat Y, HEfloat Z, const HEfloat* LBS_WEIGHTS): h(H), p(X, Y, Z), n(0.0f, 0.0f, 0.0f) {
        for (int i = 0; i < N_SMPL_JOINTS; i++)
            lbs_weights[i] = LBS_WEIGHTS[i];
    }
};

class HEMesh::MeshStruct {
  protected:
    std::vector<HEMesh::Vertex> verts;
    std::vector<HEMesh::Face> faces;
    std::vector<HEMesh::HEdge> hedges;

  public:
    // constructors
    MeshStruct() {}
    MeshStruct(const MeshStruct & mesh);
    MeshStruct(const char * obj_fn);
    MeshStruct(const std::vector<HEfloat> & verts, const std::vector<HEint> & faces);
    
    // operators
    MeshStruct operator=(const MeshStruct & obj_fn);
    
    // set
    void setFromOBJ(const char * obj_fn);
    void setFromVertsFaces(const std::vector<HEfloat> & verts, const std::vector<HEint> & faces);
    void setFromVertsFacesWithAdditionalVertAttr(const std::vector<HEfloat> & in_verts, const std::vector<HEint> & in_faces, const std::vector<HEfloat> & in_weights);
    bool exportVertsAndAttrs(const char * fn) const;
    
    // get
    size_t n_verts() const { return verts.size(); }
    size_t n_faces() const { return faces.size(); }
    size_t n_hedges() const { return hedges.size(); }
    size_t n_edges() const { return hedges.size() / 2; }
    size_t n_edges_euler() const { return verts.size() + faces.size() - 2; }

    // computations
    void computeVertexNormals();
    void computeFaceNormals();
    void computeComputeLBSGradients(HEint joint_id);

    size_t n_triangles() { return faces.size(); }
    // display
    void print();
};


#endif
