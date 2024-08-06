#include "HEMath.h"
#include "HEMesh.h"
#include <iostream>

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Please provide three arguments:\n"
                  << "  1. the path to the canonically posed SMPL template (obj format)\n"
                  << "  2. the path to the lbs_weights (txt format)\n"
                  << "  3. the path to output file (txt format)\n";
        return 0;
    }

    std::vector<HEfloat> verts;
    std::vector<HEint> faces;
    std::vector<HEfloat> weights;

    bool data_loaded = HEMesh::loadFromOBJ(argv[1], verts, faces) && HEMesh::loadWeights(argv[2], weights);
    if (data_loaded) {

        HEMesh::MeshStruct mesh;
        mesh.setFromVertsFacesWithAdditionalVertAttr(verts, faces, weights);
        mesh.computeVertexNormals();
        for (int jid = 0; jid < N_SMPL_JOINTS; jid++) {
            mesh.computeComputeLBSGradients(jid);
        }

        bool grads_computed = mesh.exportVertsAndAttrs(argv[3]);
        if (grads_computed) {
            std::cout << "Outputted to: " << argv[3] << "\n";
        }
        // mesh.exportVertsAndAttrs("cano_data_with_lbs_grad.txt");
    }
}