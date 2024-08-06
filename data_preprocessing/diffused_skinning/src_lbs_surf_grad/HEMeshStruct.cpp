#include "HEMesh.h"

#include <iostream>
#include <fstream>

#include <unordered_set>    // for edge lookup
#include <cstdio>
#include <cstring>

struct HEdgeForLookUp { // only temporarily used for initilization
    HEint v_from;
    HEint v_to;
    HEint h_idx;

    HEdgeForLookUp() { v_from = v_to = h_idx = -1; }
    HEdgeForLookUp(HEint in_v_from, HEint in_v_to, HEint in_h_idx) {
        v_from = in_v_from; v_to = in_v_to; h_idx = in_h_idx;
    }
    bool operator==(const HEdgeForLookUp & other) const {
        // check v_idx only
        if (v_from == other.v_from && v_to == other.v_to)
            return true;
        return false;
    }
};

namespace std {
    template<> struct hash<HEdgeForLookUp> {
        uint64_t operator()(const HEdgeForLookUp & h) const noexcept {
            return (uint64_t)h.v_from << 32 | (uint64_t)h.v_to;
        }
    };
}

HEMesh::MeshStruct::MeshStruct(const MeshStruct & other) {
    verts = other.verts;
    faces = other.faces;
    hedges = other.hedges;
}

HEMesh::MeshStruct::MeshStruct(const char * obj_fn) {
    setFromOBJ(obj_fn);
}

HEMesh::MeshStruct::MeshStruct(const std::vector<HEfloat> & in_verts, const std::vector<HEint> & in_faces) {
    setFromVertsFaces(in_verts, in_faces);
}

bool HEMesh::MeshStruct::exportVertsAndAttrs(const char* fn) const {
    std::ofstream fout(fn);
    if (!fout.is_open()) {
        std::cerr << "cannot open the file for output at\n  " << fn << "\ncheck the path\n";
        return false;
    }
    for (HEint vid = 0; vid < n_verts(); vid++) {
        fout << verts[vid].p[0] << " " << verts[vid].p[1] << " " << verts[vid].p[2] << " ";     // coords
        fout << verts[vid].n[0] << " " << verts[vid].n[1] << " " << verts[vid].n[2] << " ";     // normal
        fout << verts[vid].tx[0] << " " << verts[vid].tx[1] << " " << verts[vid].tx[2] << " ";     // tangent
        fout << verts[vid].ty[0] << " " << verts[vid].ty[1] << " " << verts[vid].ty[2] << " ";     // tangent
        for (int jid = 0; jid < N_SMPL_JOINTS; jid++) {
            fout << verts[vid].lbs_weights[jid] << " ";     // lbs_weights
        }
        for (int jid = 0; jid < N_SMPL_JOINTS; jid++) {
            fout << verts[vid].lbs_weights_grad_tx[jid] << " ";     // lbs_weights_grad
        }
        for (int jid = 0; jid < N_SMPL_JOINTS; jid++) {
            fout << verts[vid].lbs_weights_grad_ty[jid] << " ";     // lbs_weights_grad
        }
        fout << "\n";
    }
    return true;
}


HEMesh::MeshStruct HEMesh::MeshStruct::operator=(const MeshStruct & other) {
    verts = other.verts;
    faces = other.faces;
    hedges = other.hedges;
    return *this;
}

void HEMesh::MeshStruct::setFromOBJ(const char * obj_fn) {
    std::vector<HEfloat> in_verts;
    std::vector<HEint> in_faces;

    HEMesh::loadFromOBJ(obj_fn, in_verts, in_faces);
    setFromVertsFaces(in_verts, in_faces);
}

void HEMesh::MeshStruct::setFromVertsFaces(const std::vector<HEfloat> & in_verts, const std::vector<HEint> & in_faces) {

    verts.clear();
    hedges.clear();
    faces.clear();
    
    size_t n_verts = in_verts.size() / 3;
    size_t n_faces = in_faces.size() / 3;

    HEint cur_h_idx = 0;
    HEint cur_v_idx = 0;
    HEint cur_f_idx = 0;

    std::unordered_set<HEdgeForLookUp> hedges_lookup;
    std::vector<HEint> in_vert_recorded_idx(n_verts, HEint(-1));

    // processing is face-centered
    for (HEint in_fid = 0; in_fid < n_faces; in_fid++) {
        // each face must be a new face
        faces.push_back(HEMesh::Face(cur_h_idx));
        
        // but each vertex may not be a new vertex,
        // we first check if it is already in our list

        // 1st hedge in this triangle (v0 -> v1)
        HEint in_vid_to = in_faces[3*in_fid + 1]; // the id of vertex in in_verts
        if (in_vert_recorded_idx[in_vid_to] == HEint(-1)) {
            in_vert_recorded_idx[in_vid_to] = cur_v_idx;
            verts.push_back(
                HEMesh::Vertex(cur_h_idx,
                        in_verts[3*in_vid_to],
                        in_verts[3*in_vid_to + 1],
                        in_verts[3*in_vid_to + 2]));
            // pairing is not done at this stage
            // printf("pushing hedge with (-1, %d, %d, %d)\n", cur_h_idx+1, cur_f_idx, cur_v_idx);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, cur_v_idx));
            cur_v_idx++;
        }
        else {
            // printf("pushing hedge with (-1, %d, %d, %d) (recorded)\n", cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]));
        }
        cur_h_idx++;

        // 2nd hedge in this triangle (v1 -> v2)
        in_vid_to = in_faces[3*in_fid + 2]; // the id of vertex in in_verts
        if (in_vert_recorded_idx[in_vid_to] == HEint(-1)) {
            in_vert_recorded_idx[in_vid_to] = cur_v_idx;
            verts.push_back(
                HEMesh::Vertex(cur_h_idx,
                        in_verts[3*in_vid_to],
                        in_verts[3*in_vid_to + 1],
                        in_verts[3*in_vid_to + 2]));
            // printf("pushing hedge with (-1, %d, %d, %d)\n", cur_h_idx+1, cur_f_idx, cur_v_idx);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, cur_v_idx));
            cur_v_idx++;
        }
        else {
            // printf("pushing hedge with (-1, %d, %d, %d) (recorded)\n", cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]));
        }
        cur_h_idx++;

        // 3rd hedge in this triangle (v2 -> v0)
        in_vid_to = in_faces[3*in_fid]; // the id of vertex in in_verts
        if (in_vert_recorded_idx[in_vid_to] == HEint(-1)) {
            in_vert_recorded_idx[in_vid_to] = cur_v_idx;
            verts.push_back(
                HEMesh::Vertex(cur_h_idx,
                        in_verts[3*in_vid_to],
                        in_verts[3*in_vid_to + 1],
                        in_verts[3*in_vid_to + 2]));
            // printf("pushing hedge with (-1, %d, %d, %d)\n", cur_h_idx-2, cur_f_idx, cur_v_idx);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx-2, cur_f_idx, cur_v_idx));
            cur_v_idx++;
        }
        else {
            // printf("pushing hedge with (-1, %d, %d, %d) (recorded)\n", cur_h_idx-2, cur_f_idx, in_vert_recorded_idx[in_vid_to]);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx-2, cur_f_idx, in_vert_recorded_idx[in_vid_to]));
        }
        cur_h_idx++;

        HEint vid_v0 = in_vert_recorded_idx[ in_faces[3*in_fid + 0] ]; // the id of vertex in this->verts
        HEint vid_v1 = in_vert_recorded_idx[ in_faces[3*in_fid + 1] ];
        HEint vid_v2 = in_vert_recorded_idx[ in_faces[3*in_fid + 2] ];

        // insert the above hedges for pairing lookups later
        hedges_lookup.insert(HEdgeForLookUp(vid_v0, vid_v1, cur_h_idx-3));
        hedges_lookup.insert(HEdgeForLookUp(vid_v1, vid_v2, cur_h_idx-2));
        hedges_lookup.insert(HEdgeForLookUp(vid_v2, vid_v0, cur_h_idx-1));

        // all vertices & hedges processed, so is this face
        cur_f_idx++;
    }

    // pairing half edges
    for (HEint hid = 0; hid < hedges.size(); hid++) {
        HEMesh::HEdge cur_hedge = hedges[hid];
        HEint v_to = cur_hedge.v;
        HEint v_from = hedges[hedges[cur_hedge.next].next].v;
        auto pair_edge = hedges_lookup.find(HEdgeForLookUp(v_to, v_from, -1));
        if (pair_edge == hedges_lookup.end()) {
            ; // Mesh may have boundaries (not implemented). So this the paired edge may not exist
        }
        else {
            hedges[hid].pair = pair_edge->h_idx;
        }
    }
}


void HEMesh::MeshStruct::setFromVertsFacesWithAdditionalVertAttr(const std::vector<HEfloat> & in_verts, const std::vector<HEint> & in_faces, const std::vector<HEfloat> & in_weights) {

    verts.clear();
    hedges.clear();
    faces.clear();
    
    size_t n_verts = in_verts.size() / 3;
    size_t n_faces = in_faces.size() / 3;

    HEint cur_h_idx = 0;
    HEint cur_v_idx = 0;
    HEint cur_f_idx = 0;

    std::unordered_set<HEdgeForLookUp> hedges_lookup;
    std::vector<HEint> in_vert_recorded_idx(n_verts, HEint(-1));

    // processing is face-centered
    for (HEint in_fid = 0; in_fid < n_faces; in_fid++) {
        // each face must be a new face
        faces.push_back(HEMesh::Face(cur_h_idx));
        
        // but each vertex may not be a new vertex,
        // we first check if it is already in our list

        // 1st hedge in this triangle (v0 -> v1)
        HEint in_vid_to = in_faces[3*in_fid + 1]; // the id of vertex in in_verts
        if (in_vert_recorded_idx[in_vid_to] == HEint(-1)) {
            in_vert_recorded_idx[in_vid_to] = cur_v_idx;
            verts.push_back(
                HEMesh::Vertex(cur_h_idx,
                        in_verts[3*in_vid_to],
                        in_verts[3*in_vid_to + 1],
                        in_verts[3*in_vid_to + 2],
                        in_weights.data() + N_SMPL_JOINTS * in_vid_to
                ));
            // pairing is not done at this stage
            // printf("pushing hedge with (-1, %d, %d, %d)\n", cur_h_idx+1, cur_f_idx, cur_v_idx);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, cur_v_idx));
            cur_v_idx++;
        }
        else {
            // printf("pushing hedge with (-1, %d, %d, %d) (recorded)\n", cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]));
        }
        cur_h_idx++;

        // 2nd hedge in this triangle (v1 -> v2)
        in_vid_to = in_faces[3*in_fid + 2]; // the id of vertex in in_verts
        if (in_vert_recorded_idx[in_vid_to] == HEint(-1)) {
            in_vert_recorded_idx[in_vid_to] = cur_v_idx;
            verts.push_back(
                HEMesh::Vertex(cur_h_idx,
                        in_verts[3*in_vid_to],
                        in_verts[3*in_vid_to + 1],
                        in_verts[3*in_vid_to + 2],
                        in_weights.data() + N_SMPL_JOINTS * in_vid_to));
            // printf("pushing hedge with (-1, %d, %d, %d)\n", cur_h_idx+1, cur_f_idx, cur_v_idx);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, cur_v_idx));
            cur_v_idx++;
        }
        else {
            // printf("pushing hedge with (-1, %d, %d, %d) (recorded)\n", cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx+1, cur_f_idx, in_vert_recorded_idx[in_vid_to]));
        }
        cur_h_idx++;

        // 3rd hedge in this triangle (v2 -> v0)
        in_vid_to = in_faces[3*in_fid]; // the id of vertex in in_verts
        if (in_vert_recorded_idx[in_vid_to] == HEint(-1)) {
            in_vert_recorded_idx[in_vid_to] = cur_v_idx;
            verts.push_back(
                HEMesh::Vertex(cur_h_idx,
                        in_verts[3*in_vid_to],
                        in_verts[3*in_vid_to + 1],
                        in_verts[3*in_vid_to + 2],
                        in_weights.data() + N_SMPL_JOINTS * in_vid_to));
            // printf("pushing hedge with (-1, %d, %d, %d)\n", cur_h_idx-2, cur_f_idx, cur_v_idx);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx-2, cur_f_idx, cur_v_idx));
            cur_v_idx++;
        }
        else {
            // printf("pushing hedge with (-1, %d, %d, %d) (recorded)\n", cur_h_idx-2, cur_f_idx, in_vert_recorded_idx[in_vid_to]);
            hedges.push_back(HEMesh::HEdge(-1, cur_h_idx-2, cur_f_idx, in_vert_recorded_idx[in_vid_to]));
        }
        cur_h_idx++;

        HEint vid_v0 = in_vert_recorded_idx[ in_faces[3*in_fid + 0] ]; // the id of vertex in this->verts
        HEint vid_v1 = in_vert_recorded_idx[ in_faces[3*in_fid + 1] ];
        HEint vid_v2 = in_vert_recorded_idx[ in_faces[3*in_fid + 2] ];

        // insert the above hedges for pairing lookups later
        hedges_lookup.insert(HEdgeForLookUp(vid_v0, vid_v1, cur_h_idx-3));
        hedges_lookup.insert(HEdgeForLookUp(vid_v1, vid_v2, cur_h_idx-2));
        hedges_lookup.insert(HEdgeForLookUp(vid_v2, vid_v0, cur_h_idx-1));

        // all vertices & hedges processed, so is this face
        cur_f_idx++;
    }
    // pairing half edges
    for (HEint hid = 0; hid < hedges.size(); hid++) {
        HEMesh::HEdge cur_hedge = hedges[hid];
        HEint v_to = cur_hedge.v;
        HEint v_from = hedges[hedges[cur_hedge.next].next].v;
        auto pair_edge = hedges_lookup.find(HEdgeForLookUp(v_to, v_from, -1));
        if (pair_edge == hedges_lookup.end()) {
            ;// Mesh may have boundaries!!! So this the paired edge may not exist!!!
        }
        else {
            hedges[hid].pair = pair_edge->h_idx;
        }
    }
}

void HEMesh::MeshStruct::computeVertexNormals() {
    // clear existing normals in vertices
    size_t n_verts = verts.size();
    for (size_t vid = 0; vid < n_verts; vid++) {
        verts[vid].n = 0.0f;
    }

    // traverse faces and accumulate normals
    // these normals are weighted by the triangle areas
    // to be normalized later
    size_t n_faces = faces.size();
    HEint hid;
    HEMath::Vec3 e0, /*e1,*/ e2;    // NOTE: e0 = v1 - v0
    HEMath::Vec3 face_n;             // NOTE: face normal (with signed area)
    for (size_t fid = 0; fid < n_faces; fid++) {
        hid = faces[fid].h;     HEint vid0 = hedges[hid].v;
        hid = hedges[hid].next; HEint vid1 = hedges[hid].v;
        hid = hedges[hid].next; HEint vid2 = hedges[hid].v;

        e0 = verts[vid1].p - verts[vid0].p;
        // e1 = verts[vid2].p - verts[vid1].p;
        e2 = verts[vid0].p - verts[vid2].p;

        face_n = HEMath::cross_prod(e0, -e2);

        verts[vid0].n = verts[vid0].n + face_n;
        verts[vid1].n = verts[vid1].n + face_n;
        verts[vid2].n = verts[vid2].n + face_n;
    }

    // normalize them (inplace)
    for (size_t vid = 0; vid < n_verts; vid++) {
        verts[vid].n.normalize_();
    }
}


void HEMesh::MeshStruct::computeFaceNormals() {
    size_t n_faces = faces.size();
    HEint hid;
    HEMath::Vec3 e0, /*e1,*/ e2;    // NOTE: e0 = v1 - v0
    HEMath::Vec3 face_n;             // NOTE: face normal (with signed area)
    for (size_t fid = 0; fid < n_faces; fid++) {
        hid = faces[fid].h;     HEint vid0 = hedges[hid].v;
        hid = hedges[hid].next; HEint vid1 = hedges[hid].v;
        hid = hedges[hid].next; HEint vid2 = hedges[hid].v;

        e0 = verts[vid1].p - verts[vid0].p;
        e2 = verts[vid0].p - verts[vid2].p;

        faces[fid].n = HEMath::cross_prod(e0, -e2);
        faces[fid].n.normalize_();
    }
}

void HEMesh::MeshStruct::computeComputeLBSGradients(HEint joint_id) {
    // set tangent frames
    size_t n_verts = verts.size();
    for (size_t vid = 0; vid < n_verts; vid++) {
        verts[vid].tx = verts[vid].n.vertical();
        verts[vid].ty = HEMath::cross_prod(verts[vid].n, verts[vid].tx);
    }

    // traverse over verts and their neighbors
    n_verts = verts.size();
    for (size_t vid = 0; vid < n_verts; vid++) {
        verts[vid].lbs_weights_grad_tx[joint_id] = 0.0f;
        verts[vid].lbs_weights_grad_ty[joint_id] = 0.0f;
        int counter_neighbor = 0;
        
        // traverse over neighbors
        HEint hid_start = verts[vid].h;
        HEint hid_cur = hid_start;
        do {
            HEint vid_other = hedges[hedges[hid_cur].pair].v;
            HEMath::Vec3 pos_diff = verts[vid_other].p - verts[vid].p;
            HEfloat val_diff = verts[vid_other].lbs_weights[joint_id] - verts[vid].lbs_weights[joint_id];
            HEfloat val_diff_normalized = val_diff / pos_diff.norm();

            verts[vid].lbs_weights_grad_tx[joint_id] += val_diff_normalized * HEMath::inner_prod(pos_diff.normalize(), verts[vid].tx);
            verts[vid].lbs_weights_grad_ty[joint_id] += val_diff_normalized * HEMath::inner_prod(pos_diff.normalize(), verts[vid].ty);

            hid_cur = hedges[hedges[hid_cur].next].pair;
            counter_neighbor++;
        } while (hid_cur != hid_start);
        verts[vid].lbs_weights_grad_tx[joint_id] /= HEfloat(counter_neighbor);
        verts[vid].lbs_weights_grad_ty[joint_id] /= HEfloat(counter_neighbor);
    }
}


void HEMesh::MeshStruct::print() {
    size_t n_hedges = hedges.size();
    size_t n_verts = verts.size();
    size_t n_faces = faces.size();
    printf("In this struct there are:\n");
    printf("%ld half edges\n", n_hedges);
    printf("%ld vertices\n", n_verts);
    printf("%ld faces\n", n_faces);
    
    printf("half edges:\n");
    for (HEint i = 0; i < n_hedges; i++) {
        printf("h[%d]: (pair = %d, next = %d, f = %d, v = %d)\n", i,
            hedges[i].pair, hedges[i].next, hedges[i].f, hedges[i].v);
    }
    printf("vertices:\n");
    for (HEint i = 0; i < n_verts; i++) {
        printf("v[%d]: (h = %d, x = %.4f, y = %.4f, z = %.4f)\n", i,
            verts[i].h, verts[i].p[0], verts[i].p[1], verts[i].p[2]);
    }
    printf("faces:\n");
    for (HEint i = 0; i < n_faces; i++) {
        printf("f[%d]: (h = %d)\n", i, faces[i].h);
    }
}

bool HEMesh::loadFromOBJ(const char * obj_fn, 
	    std::vector<HEfloat> & out_verts, 
	    std::vector<HEint> & out_v_indices){
	
    printf("Loading obj file %s...\n", obj_fn);

	std::vector<HEint> vertexIndices;
	std::vector<HEfloat> temp_vertices;


	FILE * file = fopen(obj_fn, "r");
	if( file == NULL ){
        std::cerr << "cannot open the file at\n  " << obj_fn << "\ncheck the path\n";
		return false;
	}

	while(true){

		char lineHeader[256];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		// else : parse lineHeader
		
		if ( strcmp( lineHeader, "v" ) == 0 ){
			HEfloat x, y, z;
			fscanf(file, "%f %f %f\n", &x, &y, &z );
			// temp_vertices.push_back(vertex);
            out_verts.push_back(x);
            out_verts.push_back(y);
            out_verts.push_back(z);
		} else if ( strcmp( lineHeader, "f" ) == 0 ){
            HEint vid0, vid1, vid2;
            
			int matches = fscanf(file, "%d %d %d\n", &vid0, &vid1, &vid2);
			if (matches != 3){
                std::cerr << "file can't be read by this parser. Are you using OBJ format with v- and f- entries only?\n";

				fclose(file);
				return false;
			}
            // NOTE!!! minus 1 for all indices!!!
            out_v_indices.push_back(vid0 - 1);
            out_v_indices.push_back(vid1 - 1);
            out_v_indices.push_back(vid2 - 1);
		} else{
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file);
		}

	}

	fclose(file);
	return true;
}

bool HEMesh::loadWeights(const char * txt_fn, std::vector<HEfloat> & out_weights) {
    printf("Loading txt file %s...\n", txt_fn);

    out_weights.clear();

    std::ifstream fin(txt_fn);
    if (!fin.is_open()) {
        std::cerr << "cannot open the file at\n  " << txt_fn << "\ncheck the path\n";
        return false;
    }

    float cur_num;

    while (fin >> cur_num) {
        out_weights.push_back(cur_num);
    }
    fin.close();
    assert(out_weights.size() == N_SMPL_JOINTS * N_SMPL_VERTS);

	return true;
}


bool HEMesh::writeToOBJ(const char * obj_fn, const std::vector<HEfloat> & out_verts, const std::vector<HEint> & out_faces) {
    std::ofstream fout(obj_fn);
    size_t n_verts = out_verts.size() / 3;
    size_t n_faces = out_faces.size() / 3;
    for (int vid = 0; vid < n_verts; vid++) {
        fout << "v " << out_verts[vid* 3 + 0]
             << " "  << out_verts[vid* 3 + 1]
             << " "  << out_verts[vid* 3 + 2]
             << "\n";
    }
    for (int fid = 0; fid < n_faces; fid++) {
        fout << "f " << out_faces[fid* 3 + 0] + 1
             << " "  << out_faces[fid* 3 + 1] + 1
             << " "  << out_faces[fid* 3 + 2] + 1
             << "\n";
    }
    return true;
}
