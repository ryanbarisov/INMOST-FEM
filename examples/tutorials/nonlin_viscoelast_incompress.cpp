// Created by Liogky Alexey on 09.02.2024.
//
/**
 * This program generates  and solves a finite element system for the stationary nonlinear incompressible viscoelasticity problem
 *
 * \f[
 * \begin{aligned}
 *   \mathrm{div}\ \mathbb{P}  + f &= 0\    in\  \Omega  \\
 *                           J - 1 &= 0\    in\  \Omega  \\
 *          \mathbf{u}             &= \mathbf{u}_0\  on\  \Gamma_D\\
 *     \mathbb{P} \cdot \mathbf{N} &= -p_{ext}\ \mathrm{adj}\ \mathbb{F}^T \mathbf{N}\ on\  \Gamma_P\\
 *     \mathbb{P} \cdot \mathbf{N} &= 0\    on\  \Gamma_0\\
 *                      \mathbb{P} &= \mathbb{F} \cdot \mathbb{S} - p \mathrm{adj}\ \mathbb{F}^T\\
 *   \mathbb{F}_{ij} &= \mathbb{I}_{ij} + \mathrm{grad}_j \mathbf{u}_i,\ J = \mathrm{det }\mathbb{F}\\
 *        \mathbb{S} &= \lambda\ \mathrm{tr}\ \mathbb{E}\ \mathbb{I} + 2 \mu \mathbb{E} \\
 *        \mathbb{E} &= \frac{\mathbb{F}^T \cdot \mathbb{F}  - \mathbb{I} } {2} \\
 * \end{aligned}
 * \f]
 *
 *
 * where where Ω = [0,10] x [0,1]^2, Γ_D = {0}x[0,1]^2, Γ_P = [0,10]x[0,1]x{0}, Γ_0 = ∂Ω \ (Γ_D ⋃ Γ_P)
 * The user-defined coefficients are
 *  μ(x)   = 3.0         - first Lame coefficient
 *  λ(x)   = 1.0         - second Lame coefficient
 *  f(x)   = { 0, 0, 0 } - external body forces
 *  u_0(x) = { 0, 0, 0 } - essential (Dirichlet) boundary condition
 *  p_{ext}(x)   = 0.001         - pressure on bottom part
 *
 */
#include "prob_args.h"
#include "anifem++/autodiff/cauchy_strain_autodiff.h"
#include "anifem++/kinsol_interface/SUNNonlinearSolver.h"
#include "soe.hpp"
using namespace INMOST;
double dT = 0.005; // time step [s]
double T = 0.0; // current time [s]
double T_FINAL = 6.0; // final time [s]
double T_SHEAR = 1.0;//time to start shearing [s]
double H = 2.7e-3;//cylinder height [m], should be consistent with mesh
double R = 10e-3;//cylinder radius [m], should be consistent with mesh
double CS = 0.1; // compression strain [percent]]
double shear_strain_ampl = 0.25;//shear strain amplitude [percent]
double freq = 1.0; //f = 1 [Hz]
double alpha = 0.2; // Fractional order of Caputo derivative
double C = 97.4; // scaling parameter of Mooney-Rivlin model [Pa]
double delta = 130.8; // scaling parameter of Mooney-Rivlin model [Pa]
//double delta = 126.7; // scaling parameter of viscoelastic exponential model [Pa]
double b = 1.5; // potential param
double rho = 1.0; // material density
std::string mesh_fname = "../../../data/mesh/viscoelastic_cylinder.msh";
std::string coef_fname = "../../../data/soe/coefs_nexp09_T00100.csv";
bool use_kinsol = false;
bool debug_soe = false;
int save_steps = 1;

std::string MODEL = "vMR";//vMR -- Mooney-Rivlin, vEXP -- Fung (exponential)

struct InputArgs1VarNonLin: public InputArgs2Var {
    using ParentType = InputArgs2Var;
    double nlin_rel_err = 1e-8, nlin_abs_err = 1e-8;
    int nlin_maxit = 10;
    double lin_abs_scale = 0.01;
    uint parseArg(int argc, char* argv[], bool print_messages = true) override {
        #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
            else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
        uint i = 0;
        if (strcmp(argv[i], "-re") == 0 || strcmp(argv[i], "--rel_error") == 0){
            GETARG(nlin_rel_err = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-ae") == 0 || strcmp(argv[i], "--abs_error") == 0){
            GETARG(nlin_abs_err = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-ni") == 0 || strcmp(argv[i], "--maxits") == 0){
            GETARG(nlin_maxit = std::stoi(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--use_kinsol") == 0){
            GETARG(use_kinsol = std::stoi(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--latol_scl") == 0){
            GETARG(lin_abs_scale = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mesh_name") == 0){
            GETARG(mesh_fname = argv[++i];)
            return i+1;
        } if (                               strcmp(argv[i], "--alpha") == 0){
            GETARG(alpha = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-cf") == 0 || strcmp(argv[i], "--coef_name") == 0){
            GETARG(coef_fname = argv[++i];)
            return i+1;
        } if (                               strcmp(argv[i], "--debug_soe") == 0){
            GETARG(debug_soe = std::stoi(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--save_steps") == 0){
            GETARG(save_steps = std::stoi(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-CS") == 0 || strcmp(argv[i], "--compression_strain") == 0){
            GETARG(CS = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-SS") == 0 || strcmp(argv[i], "--shear_strain") == 0){
            GETARG(shear_strain_ampl = std::stod(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--freq") == 0){
            GETARG(freq = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "--time_step") == 0){
            GETARG(dT = std::stod(argv[++i]);)
            return i+1;
        } if (                              strcmp(argv[i], "--time_shear") == 0){
            GETARG(T_SHEAR = std::stod(argv[++i]);)
            return i+1;
        } if (                              strcmp(argv[i], "--time_final") == 0){
            GETARG(T_FINAL = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--height") == 0){
            GETARG(H = std::stod(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--b") == 0){
            GETARG(b = std::stod(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--C") == 0){
            GETARG(C = std::stod(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--delta") == 0){
            GETARG(delta = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-rho") == 0 || strcmp(argv[i], "--density") == 0){
            GETARG(rho = std::stod(argv[++i]);)
            return i+1;
        } if (                               strcmp(argv[i], "--model") == 0){
            GETARG(MODEL = argv[++i];)
            return i+1;
        } else
            return ParentType::parseArg(argc, argv, print_messages);
        #undef GETARG
    }
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override{
        out << prefix << "newton: stop tolerances: rel_tol = " << nlin_rel_err << ", abs_tol = " << nlin_abs_err << "\n";
        out << prefix << "newton: maximum iteration number = " << nlin_maxit << "\n";
        out << prefix << "newton: use KINSOL to solve nonlinear system = " << use_kinsol << "\n";
        out << prefix << "newton: linear solver absolute tolerance scale = " << lin_abs_scale << "\n";
        out << prefix << "model : mesh file name = " << mesh_fname << "\n";
        out << prefix << "model : (debug) use ordinary derivative = " << debug_soe << "\n";
        out << prefix << "model : fractional Caputo order = " << alpha << "\n";
        out << prefix << "model : file name of SoE coefficients table = " << coef_fname << "\n";
        out << prefix << "model : save mesh every N steps, N = " << save_steps << "\n";
        out << prefix << "model : compression strain = " << CS << "\n";
        out << prefix << "model : shear strain = " << shear_strain_ampl << "\n";
        out << prefix << "model : frequency = " << freq << "\n";
        out << prefix << "model : time step = " << dT << "\n";
        out << prefix << "model : time to shear = " << T_SHEAR << "\n";
        out << prefix << "model : time to finish = " << T_FINAL << "\n";
        out << prefix << "model : cylinder height = " << H << "\n";
        out << prefix << "model : vEXP scaling parameter (b) = " << b << "\n";
        out << prefix << "model : elastic scaling parameter (C) = " << C << "\n";
        out << prefix << "model : viscoelastic scaling parameter (delta) = " << delta << "\n";
        out << prefix << "model : material density (rho) = " << rho << "\n";
        out << prefix << "model : model name (MODEL) = " << MODEL << "\n";
        ParentType::print(out, prefix);
    }
protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override{
        out << prefix << "  -re, --rel_error  DVAL    <Set stop relative residual norm for newton method, default=\"" << nlin_rel_err << "\">\n";
        out << prefix << "  -ae, --abs_error  DVAL    <Set stop absolute residual norm for newton method, default=\"" << nlin_abs_err << "\">\n";
        out << prefix << "  -ni, --maxits     IVAL    <Set maximum number of newton method iterations, default=\"" << nlin_maxit << "\">\n";
        out << prefix << "       --use_kinsol IVAL    <Whether to use KINSOL to solve nonlinear system, default=\"" << use_kinsol << "\">\n";
        out << prefix << "       --latol_scl  DVAL    <Set newton linear solver absolute tolerance scale, default=\"" << lin_abs_scale << "\">\n";
        out << prefix << "  -m,  --mesh_name  SVAL    <Specify mesh file name, default=\"" << mesh_fname << "\">\n";
        out << prefix << "       --debug_soe  IVAL    <Whether to use ordinary derivative instead of fractional Caputo derivative, default=\"" << debug_soe << "\">\n";
        out << prefix << "       --alpha      DVAL    <Set order of fractional Caputo derivative, default=\"" << alpha << "\">\n";
        out << prefix << "  -cf, --coef_name  SVAL    <Specify file name for sum-of-exponents coefficients, default=\"" << coef_fname << "\">\n";
        out << prefix << "       --save_steps IVAL    <Save mesh after this number of step, default=\"" << save_steps << "\">\n";
        out << prefix << " MODEL PARAMETERS\n";
        out << prefix << "  -CS, --compression_strain  DVAL    <Set compression strain (percent), default=\"" << CS << "\">\n";
        out << prefix << "  -SS, --shear_strain        DVAL    <Set shear strain (percent), default=\"" << shear_strain_ampl << "\">\n";
        out << prefix << "       --freq       DVAL    <Set frequency, default=\"" << freq << "\">\n";
        out << prefix << "  -dt, --time_step  DVAL    <Set time step, default=\"" << dT << "\">\n";
        out << prefix << "       --time_shear DVAL    <Set time of shear start, default=\"" << T_SHEAR << "\">\n";
        out << prefix << "       --time_final DVAL    <Set finish time, default=\"" << T_FINAL << "\">\n";
        out << prefix << "  -H,  --height     DVAL    <Set cylinder height (for correct BC setup), default=\"" << H << "\">\n";
        out << prefix << "       --b          DVAL    <Set vEXP scaling parameter, default=\"" << b << "\">\n";
        out << prefix << "       --C          DVAL    <Set elastic scaling parameter, default=\"" << C << "\">\n";
        out << prefix << "       --delta      DVAL    <Set viscoelastic scaling parameter, default=\"" << delta << "\">\n";
        out << prefix << "-rho,  --density    DVAL    <Set material density, default=\"" << rho << "\">\n";
        out << prefix << "       --model      SVAL    <Set model name (vMR - Mooney-Rivlin, vEXP - Fung exponential), default=\"" << MODEL << "\">\n";
        ParentType::printArgsDescr(out, prefix);
    }
};
/// Reorder in-place quadrature data stored relative old_node_enumerator to be stored relative new_node_enumerator
void reorder_quadrature_function(Tag t, int quad_order, INMOST::Tag old_node_enumerator, INMOST::Tag new_node_enumerator, unsigned func_dim){
    auto formula = tetrahedron_quadrature_formulas(quad_order);
    assert(formula.IsSymmetric() && "reorder_quadrature_function currently support only symmetric quadraure point distributions");
    assert(old_node_enumerator.isDefined(INMOST::NODE) && new_node_enumerator.isDefined(INMOST::NODE) && "Wrong node enumerator");
    auto _csym = formula.GetSymmetryPartition();
    std::array<Ani::DofT::uint, 5> csym;
    for (int i = 0; i < 5; ++i) csym[i] = static_cast<Ani::DofT::uint>(_csym[i]);
    Ani::DofT::DofSymmetries sym({0}, {0}, {0}, {0}, {0}, csym);
    Ani::DofT::UniteDofMap dmap(sym);
    Ani::reorder_mesh_function_data({t}, dmap, old_node_enumerator, new_node_enumerator, func_dim);
}
/// Rearrange cell_local_data, which is ordered properly, to be ordered relative to node_ids order
template<typename iterator1, typename iterator2>
void reordered_copy_data_of_quadrature_function(std::array<long, 4> node_ids, int quad_order, iterator1 cell_local_data, iterator2 to, unsigned func_dim){
    auto formula = tetrahedron_quadrature_formulas(quad_order);
    assert(formula.IsSymmetric() && "reordered_copy_data_of_quadrature_function currently support only symmetric quadraure point distributions");
    auto sym = formula.GetSymmetryPartition();
    bool comp_node_perm = ((abs(sym[1]) + abs(sym[2]) + abs(sym[3]) + abs(sym[4])) != 0);
    std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
    if (comp_node_perm)
        canonical_node_indexes = Ani::createOrderPermutation(node_ids.data());
    std::size_t shift = 0, sz = func_dim;
    for (unsigned ist = 0; ist < sym.size(); ++ist) if (sym[ist] > 0){
        auto vol = Ani::DofT::DofSymmetries::symmetry_volume(Ani::DofT::CELL, ist);
        for (int lsid = 0; lsid < vol; ++lsid){
            auto reordered_lsid = Ani::DofT::DofSymmetries::index_on_reorderd_elem(Ani::DofT::CELL, 0, ist, lsid, canonical_node_indexes.data());
            for (int is = 0; is < sym[ist]; ++is)
                std::copy(cell_local_data + shift + sz*(is*vol + reordered_lsid), cell_local_data + shift + sz*(is*vol + reordered_lsid+1) , to + shift + sz*(is*vol + lsid));
        }
        shift += sz * sym[ist] * vol;
    }
}
/// Copy data from 'from' to 'to' changing ordering of nodes from proper (relative GlobalID) to custom (same as nodes ordering in 'nds')
void copy_quadrature_data_from_proper_to_custom_order(const double* from, double* to, int quad_order, const Mesh* mlink, const INMOST::HandleType* nds/*[4]*/, unsigned func_dim){
    std::array<long, 4> gni;
    for (int i = 0; i < 4; ++i)
        gni[i] = INMOST::Node(const_cast<INMOST::Mesh*>(mlink), nds[i]).GlobalID();
    reordered_copy_data_of_quadrature_function(gni, quad_order, from, to, func_dim);
}
/// Copy data from 'from' to 'to' changing ordering of nodes from custom (same as nodes ordering in 'nds')  to proper (relative GlobalID)
void copy_quadrature_data_from_custom_to_proper_order(const double* from, double* to, int quad_order, const Mesh* mlink, const INMOST::HandleType* nds/*[4]*/, unsigned func_dim){
    std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
    std::array<long, 4> gni;
    for (int i = 0; i < 4; ++i)
        gni[i] = INMOST::Node(const_cast<INMOST::Mesh*>(mlink), nds[i]).GlobalID();
    canonical_node_indexes = Ani::createOrderPermutation(gni.data());
    std::array<long, 4> rev_ids;
    for (int i = 0; i < 4; ++i)
        rev_ids[canonical_node_indexes[i]] = i;
    reordered_copy_data_of_quadrature_function(rev_ids, quad_order, from, to, func_dim);
}
inline double heaviside(double x) {return x>=0.? 1.0 : 0.0 ;}
inline double t_hat(double t) {return std::max(0.0, t-T_SHEAR);}
inline double lambda(double t) {return 1 - CS*std::min(t, T_SHEAR);}
inline double psi(double t) {return H/R*shear_strain_ampl * pow(lambda(t),1.5)*sin(2*M_PI*freq*t_hat(t));}
inline double Psi(const Coord<>& XYZ, double t) {return psi(t)*XYZ[2]/H;}
inline double position(const Coord<>& XYZ, double t, int i)
{
    if(i == 0) return 1./sqrt(lambda(t)) * (XYZ[0]*cos(Psi(XYZ,t))-XYZ[1]*sin(Psi(XYZ,t)));
    if(i == 1) return 1./sqrt(lambda(t)) * (XYZ[0]*sin(Psi(XYZ,t))+XYZ[1]*cos(Psi(XYZ,t)));
    return XYZ[2]*lambda(t);
}
PhysArr<3> displacement(const Coord<>& X, double T)
{
    return PhysArr<3>({position(X,T,0)-X[0], position(X,T,1)-X[1], position(X,T,2)-X[2]});
}
/*inline double dt_hat(double t) {return heaviside(t-T_SHEAR);}
inline double dlambda(double t) {return -CS*heaviside(T_SHEAR-t);}
inline double dpsi(double t)
{
    return H/R*shear_strain_ampl *
    (
        1.5*sqrt(lambda(t))*dlambda(t) * sin(2*M_PI*freq*t_hat(t)) +
        pow(lambda(t),1.5) * 2*M_PI*freq*cos(2*M_PI*freq*t_hat(t)) * dt_hat(t)
    );
}
inline double dPsi(const Coord<>& XYZ, double t) {return dpsi(t)*XYZ[2]/H;}
inline double velocity(const Coord<>& XYZ, double t, int i)
{
    if(i == 0) return -0.5*dlambda(t)/lambda(t)*position(XYZ,t,0) - dPsi(XYZ,t)*position(XYZ,t,1);
    if(i == 1) return -0.5*dlambda(t)/lambda(t)*position(XYZ,t,1) + dPsi(XYZ,t)*position(XYZ,t,0);
    return XYZ[2]*dlambda(t);
}*/
int main(int argc, char* argv[]){
    InputArgs1VarNonLin p;
    std::string prob_name = "nonlin_viscoelast_incompress";
    p.save_prefix = prob_name + "_out";

    p.parseArgs_mpi(&argc, &argv);
    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);
    if(pRank == 0) p.print(std::cout, "\t");

    Mesh* m = new Mesh;
    //m->SetFileOption("VERBOSITY","2");
    if(m->isParallelFileFormat(mesh_fname))
    {
        if (pRank == 0) std::cout << "Read mesh from parallel file \"" << mesh_fname << "\"" << std::endl;
        m->Load(mesh_fname);
    }
    else if(pRank == 0)
    {
        std::cout << "Read mesh from sequential file \"" << mesh_fname << "\"" << std::endl;
        m->Load(mesh_fname);
    }
    {   //remove internal tags if it have
        std::vector<std::string> tmp_tag_names;
        m->ListTagNames(tmp_tag_names);
        for (const auto& name: tmp_tag_names)
            if ((name.size() > 0 && name[0] == '_') || (name.size() > 18 && name.substr(0, 18) == "IGlobEnumeration::"))
            m->DeleteTag(m->GetTag(name));
    }
    RepartMesh(m,true);
    m->AssignGlobalID(NODE|EDGE|FACE|CELL);
    if(pCount > 1) m->ExchangeGhost(1,NODE);
    print_mesh_sizes(m);

    using namespace Ani;
    //generate FEM space from it's name
    FemSpace UFem = choose_space_from_name(p.USpace)^3;
    FemSpace PFem = choose_space_from_name(p.PSpace);
    uint unf = UFem.dofMap().NumDofOnTet(), pnf = PFem.dofMap().NumDofOnTet();
    uint quad_order = p.max_quad_order;
    uint nquad = tetrahedron_quadrature_formulas(quad_order).GetNumPoints();
    auto umask = GeomMaskToInmostElementType(UFem.dofMap().GetGeomMask());
    auto pmask = GeomMaskToInmostElementType(PFem.dofMap().GetGeomMask());
    auto mask = umask | FACE;
    
    // Set boundary labels on all boundaries
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, mask, NONE, 1);
    auto bmrk = m->CreateMarker();
    m->MarkBoundaryFaces(bmrk);
    const int INTERNAL_PART = 0, FREE_BND = 1, DIRICHLET_BND = 2, DIRICHLET_BND_TOP = 4;
    for (auto it = m->BeginElement(mask); it != m->EndElement(); ++it) it->Integer(BndLabel) = INTERNAL_PART;
    for (auto it = m->BeginFace(); it != m->EndFace(); ++it) if (it->GetMarker(bmrk)) {
        std::array<double, 3> c;
        it->Centroid(c.data());
        int lbl = FREE_BND;
        if (abs(c[2] - 0) < 10*std::numeric_limits<double>::epsilon()) lbl = DIRICHLET_BND;
        if (abs(c[2] - ::H) < 10*std::numeric_limits<double>::epsilon()) lbl = DIRICHLET_BND | DIRICHLET_BND_TOP;
        auto set_label = [BndLabel](const auto& elems, int lbl){
            for (unsigned ni = 0; ni < elems.size(); ni++)
                elems[ni].Integer(BndLabel) |= lbl;
        };
        if (mask & NODE) set_label(it->getNodes(),lbl);
        if (mask & EDGE) set_label(it->getEdges(),lbl);
        if (mask & FACE) set_label(it->getFaces(),lbl);
    }

    // Structure below stores fractional derivative discretization using reduced Prony series
    // debug_soe = 1 results in ordinary derivative
    soe_data SoE_coefs = read_soe_coefs(coef_fname, alpha, debug_soe);
    // Define tags to store result
    Tag u  = createFemVarTag(m, *UFem.dofMap().target<>(), "u");//displacement(n+1)
    Tag um1 = createFemVarTag(m, *UFem.dofMap().target<>(), "uprev");//displacement(n)
    Tag um2 = createFemVarTag(m, *UFem.dofMap().target<>(), "upprev");//displacement(n-1)
    Tag p_tag = createFemVarTag(m, *PFem.dofMap().target<>(), "p");//pressure(n+1)
    TagRealArray tag_Sv = m->CreateTag("Sv", DATA_REAL, CELL, NONE, 24);// (debug) mean values over cell: Cauchy stress SIGMA (9), deformation gradient F(9), viscous stress Sv(6)
    TagRealArray tag_DSv = m->CreateTag("DSv", DATA_REAL, CELL, NONE, 6*nquad); // (debug) D_alpha(Sv) in each quadrature opint
    std::vector<Tag> up{u, p_tag}, upp{u,um1,um2,p_tag};
    // storage for viscous stresses Sv^(n+1), Sv^n and Prony series variables Q (nexp 3x3 symm.matrices)
    int nexp = SoE_coefs.nexp;
    TagRealArray tag_SvQ = m->CreateTag("tag_SvQ", DATA_REAL, CELL, NONE, 6*(nexp+2)*nquad);
    // Setup initial conditions
    {
        for(unsigned v = 0; v < upp.size(); ++v)
        {
            ElementType mmask = upp[v] == p_tag ? pmask : umask;
            for (auto it = m->BeginElement(mmask); it != m->EndElement(); ++it)
                std::fill(it->RealArray(upp[v]).begin(), it->RealArray(upp[v]).end(), 0.0);
        }
        // SvQ(t=0) = 0
        for(auto it = m->BeginCell(); it != m->EndCell(); ++it)
            std::fill(tag_SvQ[*it].begin(), tag_SvQ[*it].end(), 0.0);
    }
    struct PotentialParams {
        double b;
        double C;
        double delta;
    };
    // potential: W = Wp + We + Wv
    // pressure potential Wp
    auto Potential_P = [](PotentialParams prm, SymMtx3D<> E, double p_coef, unsigned char dif = 2){
        (void) prm;
        return -p_coef*(Mech::J<>{dif, E} - 1);
    };
    // elastic potential We
    auto Potential_E = [](PotentialParams prm, SymMtx3D<> E, unsigned char dif = 2) {
        (void) prm;
        auto I1 = Mech::I1<>{dif, E}; //I1(C) = tr(C) = C:I
        auto I2 = Mech::I2<>{dif, E}; //I2(C) = (tr(C)^2 - tr(C^2)) / 2
        auto J = Mech::J<>{dif, E}; // J(C) = det(F) in terms of E=(C-I)/2
        auto II = I1*I1 - 2*I2;// II(C) = C:C = tr(C^2) = I1*I1 - 2*I2
        // Mooney-Rivlin, quadratic term : (II(C_dev)-3)^2 / 8
        // F_dev = J^(-1/3) F, C_dev = F_dev^T F_dev = J^(-2/3) C
        // II(C_dev) = tr(C_dev^2) = J^(-4/3) tr(C^2) = J^(-4/3) II(C)
        auto W_MR2 = sq(pow(J,-4./3)*II - 3) / 8.0;
        return W_MR2;
    };
    // viscous potential Wv
    auto Potential_V = [](PotentialParams prm, SymMtx3D<> E, unsigned char dif = 2) {
        (void) prm;
        auto I1 = Mech::I1<>{dif, E}; //I1(C) = tr(C) in terms of E=(C-I)/2
        auto J = Mech::J<>{dif, E}; // J(C) = det(F) in terms of E=(C-I)/2
        if(::MODEL == "vMR")
        {
            // Mooney-Rivlin, linear neo-Hookean term : (I1(C_dev)-3) / 2
            // F_dev = J^(-1/3) F, C_dev = F_dev^T F_dev = J^(-2/3) C
            // I1(C_dev) = tr(C_dev) = J^(-2/3) tr(C) = J^(-2/3) I1
            auto W_MR1 = (pow(J,-2./3)*I1 - 3) / 2.0;
            return W_MR1;
        }
        else if(::MODEL == "vEXP") // Fung model
        {
            Param<> b(prm.b);
            auto I2 = Mech::I2<>{dif, E};
            auto II = I1*I1 - 2*I2;// II = C:C = tr(C^2) = I1*I1 - 2*I2
            // Exponential Fung-type potential model: W = (exp(b(II_C-3))-1)/(4b)
            // PK2 tensor: S_v = exp(b(II_E-3)) C 
            auto W_fung = (exp(b*(II-3)) - 1) / (4*b);// here E param is passed as distortional: det E = 1
            return W_fung;
        }
        else exit(-1);
    };
    // second Piola-Kirchoff stress S_ab = dW/dE_ab 
    // this is more complicated due to fractional derivative term
    auto S_func = [Potential_P,Potential_V,Potential_E](PotentialParams prm, const Mtx3D<>& grU, double p_coef, double* svq, const soe_data& SoE_coefs) -> SymMtx3D<> {
        SymMtx3D<> E = Mech::grU_to_E(grU), C = 2*E+SymMtx3D<>::Identity();
        double invJ23 = pow((Mtx3D<>::Identity()+grU).Det(), -2./3);//J^(-2/3)
        SymMtx3D<> E0 = (invJ23*C-SymMtx3D<>::Identity())/2;
        SymMtx3D<> Sv = Potential_V(prm, ::MODEL=="vEXP" ? E0 : E, 1).D(); // dWv/dE
        SymMtx3D<> D_alpha_Sv = SoE_coefs.get_deriv(dT, Sv, svq);
        if(::MODEL == "vEXP") // Fung model:
            D_alpha_Sv = D_alpha_Sv - 1./3 * D_alpha_Sv.Dot(C) * C.Inv(); // Dev[D_alpha(Sv)]
        else invJ23 = 1.0;
        return prm.C*Potential_E(prm,E,1).D() + prm.delta*invJ23*D_alpha_Sv + Potential_P(prm,E,p_coef,1).D();
    };
    // first Piola-Kirchoff stress P_ab = F*S_ab
    auto P_func = [S_func](PotentialParams prm, const Mtx3D<>& grU, double p_coef, double* svq, const soe_data& SoE_coefs) -> Mtx3D<> { return Mech::S_to_P(grU, S_func(prm, grU, p_coef, svq, SoE_coefs)); };
    auto MaterialElasticityTensor = [](double J, const SymMtx3D<>& C, const SymMtx3D<>& S, const BiSym4Tensor3D<>& CC) -> BiSym4Tensor3D<> {
        // CC_ijkl = d^2 Wv/(dEij dEkl)
        const SymMtx3D<> invC = C.Inv();
        const SymMtx3D<> CC_C = CC.Mul(C); // CC_ijab Cab = CC_abij Cab
        double C_CC_C = C.Dot(CC_C); // C_ab CC_abcd C_cd
        BiSym4Tensor3D<> Cs1 = BiSym4Tensor3D<>::TensorSymMul2(CC_C, invC), Cs2;
        for (auto it = Cs1.begin(); it != Cs1.end(); ++it) {
            auto q = it.index();
            *it -= 1./3 * C_CC_C * invC(q.i,q.j)*invC(q.k,q.l);
        }
        for (auto it = Cs2.begin(); it != Cs2.end(); ++it) {
            auto q = it.index();
            *it = (invC(q.i,q.k)*invC(q.j,q.l) + invC(q.i,q.l)*invC(q.j,q.k))/2
                + 1./3*invC(q.i,q.j)*invC(q.k,q.l);
        }
        Cs2 = C.Dot(S)*Cs2 - BiSym4Tensor3D<>::TensorSymMul2(S, invC);
        return pow(J,-4./3)*(CC - 1./3*Cs1 + 2./3*Cs2);
    };
    // dS_ab/dE_cd = d2W/(dE_ab dE_cd)
    // this is more complicated due to fractional derivative term
    auto dS_func = [Potential_P,Potential_V,Potential_E,MaterialElasticityTensor]
        (PotentialParams prm, const Mtx3D<>& grU, double p_coef, double* svq, const soe_data& SoE_coefs) -> BiSym4Tensor3D<> {
        SymMtx3D<> E = Mech::grU_to_E(grU), C = 2*E+SymMtx3D<>::Identity();
	
        BiSym4Tensor3D<> CC;
        if(::MODEL == "vEXP") // Fung model:
        {
            // make distortional for Fung elasticity
            double J = Mech::J<>{0, E}(), invJ23 = pow(J,-2./3);
            SymMtx3D<> E0 = (invJ23*C-SymMtx3D<>::Identity())/2;
            auto Wv = Potential_V(prm,E0,2);
            BiSym4Tensor3D<> dSv = SoE_coefs.get_coef(dT)*Wv.DD();// dSv <-- D_alpha(d2Wv/dE2)
            SymMtx3D<> Sv = Wv.D(); // dWv/dE
            SymMtx3D<> D_alpha_Sv = SoE_coefs.get_deriv(dT, Sv, svq);
            CC = MaterialElasticityTensor(J, invJ23*C, D_alpha_Sv, dSv);
        }
        else if(::MODEL == "vMR") //Mooney-Rivlin model:
            CC = SoE_coefs.get_coef(dT)*Potential_V(prm,E,2).DD();
        else exit(-1);
        return prm.C*Potential_E(prm,E,2).DD() + prm.delta*CC + Potential_P(prm,E,p_coef,2).DD();
    };
    // construct dP_ij/dF_kl using (dS/dE)_abcd = d2W/(dE_ab dE_cd), W -- Potential
    auto dP_func = [S_func,dS_func](PotentialParams prm, const Mtx3D<>& grU, double p_coef, double* svq, const soe_data& SoE_coefs) -> Sym4Tensor3D<>
    {
        return Mech::dS_to_dP(grU,
            S_func(prm,grU,p_coef,svq,SoE_coefs),
            dS_func(prm,grU,p_coef,svq,SoE_coefs));
    };
    auto dJ_func = [](const Mtx3D<>& grU)->Mtx3D<> {
        return Mech::S_to_P(grU, Mech::J<>(1, Mech::grU_to_E(grU)).D());
    };
    auto comp_gradU = [gradUFEM = UFem.getOP(GRAD)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> udofs, DynMem<>& alloc)->Mtx3D<> {
        Mtx3D<> grU;
        DenseMatrix<> A(grU.m_dat.data(), 9, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(udofs.data, udofs.size, 1), gradUFEM, A, alloc);
        return grU;
    };
    auto comp_lagrange_coef = [idenPFEM = PFem.getOP(IDEN)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> pdofs, DynMem<>& alloc)->double {
        double p = NAN;
        DenseMatrix<> pa(&p, 1, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(pdofs.data, pdofs.size, 1), idenPFEM, pa, alloc);
        return p;
    };
    auto W_params = [](const Cell& c, const Coord<> &X) -> PotentialParams {
        (void) c, (void) X;
        PotentialParams r;
        r.b = ::b;
        r.C = ::C;
        r.delta = ::delta;
        return r;
    };
    struct BndMarker{
        std::array<int, 4> n = {0};
        std::array<int, 6> e = {0};
        std::array<int, 4> f = {0};
        DofT::TetGeomSparsity getSparsity(int type) const {
            DofT::TetGeomSparsity sp;
            for (int i = 0; i < 4; ++i) if (n[i] & type)
                sp.setNode(i);
            for (int i = 0; i < 6; ++i) if (e[i] & type)
                sp.setEdge(i);
            for (int i = 0; i < 4; ++i) if (f[i] & type)
                sp.setFace(i);
            return sp;
        }
        void fillFromBndTag(Tag lbl, Ani::DofT::uint geom_mask, const ElementArray<Node>& nodes, const ElementArray<Edge>& edges, const ElementArray<Face>& faces) {
            if (geom_mask & DofT::NODE){
                for (unsigned i = 0; i < n.size(); ++i)
                    n[i] = nodes[i].Integer(lbl);
            }
            if (geom_mask & DofT::EDGE){
                for (unsigned i = 0; i < e.size(); ++i)
                    e[i] = edges[i].Integer(lbl);
            }
            if (geom_mask & DofT::FACE){
                for (unsigned i = 0; i < f.size(); ++i)
                    f[i] = faces[i].Integer(lbl);
            }
        }
    };
    struct ProbLocData {
        BndMarker lbl;      //< save labels used to apply boundary conditions
        ArrayView<> udofs;  //< save elemental dofs to evaluate grad_j u_i (x)
        ArrayView<> um1dofs;
        ArrayView<> um2dofs;
        ArrayView<> pdofs;  //< save elemental dofs to evaluate Lagrange coefficient p(x)
        ArrayView<> svqdofs;//< save elemental dofs to evaluate fractional derivative
        double rho;
        //some helper data to be postponed to tensor functions
        const Tetra<const double>* pXYZ = nullptr;
        DynMem<>* palloc = nullptr;
        const Cell* c = nullptr;
        soe_data SoE_coefs;
    };
    auto getU = [idenUFem = UFem.getOP(IDEN)](const Coord<> &X, void *user_data, int i) {
        std::array<double, 3> Up;
        auto& dat = *static_cast<ProbLocData*>(user_data);
        DenseMatrix<> U(Up.data(), Up.size(), 1);
        ArrayView<>* dofs[3] = {&dat.um2dofs, &dat.um1dofs, &dat.udofs};
        fem3DapplyX<>(*dat.pXYZ, ArrayView<const double>(X.data(),3), DenseMatrix<>(dofs[2+i]->data, dofs[2+i]->size, 1), idenUFem, U, *dat.palloc);
        return Up;
    };
    auto mass_rhs_tensor = [getU,comp_gradU](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iQuad) {
        assert(dims.first == 3 && dims.second == 1 && "Wrong mass_rhs");
        (void) dims; (void) iQuad;
        auto u = getU(X, user_data, 0), um1 = getU(X, user_data, -1), um2 = getU(X, user_data, -2);
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto J = Mech::J<>{0, Mech::grU_to_E(grU)}();
        for (unsigned i = 0; i < u.size(); ++i)
            D[i] = p.rho*(J*u[i] - (J+1)*um1[i] + um2[i])/(dT*dT);
        return TENSOR_GENERAL;
    };
    auto mass_mtx_tensor = [comp_gradU](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) X; (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto J = Mech::J<>{0, Mech::grU_to_E(grU)}();
        D[0] = D[4] = D[8] = p.rho*J/(dT*dT);
        return Ani::TENSOR_GENERAL;
    };
    auto P_tensor = [Potential_V, comp_gradU, comp_lagrange_coef, P_func, W_params, tag_Sv, tag_DSv, nquad](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iQuad){
        (void) dims;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double p_coef = comp_lagrange_coef(X, *p.pXYZ, p.pdofs, *p.palloc);
        PotentialParams c = W_params(*p.c, X);
        int offset = 6*(p.SoE_coefs.nexp+2)*iQuad;
        auto P = P_func(c, grU, p_coef, p.svqdofs.begin()+offset, p.SoE_coefs);
        std::copy(P.m_dat.data(), P.m_dat.data() + 9, D);
        SymMtx3D<> Sv = Potential_V(c,Mech::grU_to_E(grU),1).D();
        SymMtx3D<> D_alpha_Sv = p.SoE_coefs.get_deriv(dT, Sv, p.svqdofs.begin()+offset);
        real_array dat_sv = tag_Sv[*p.c];
        real_array dat_dsv = tag_DSv[*p.c];
        {
            Mtx3D<> F = Mtx3D<>::Identity() + grU;
            Mtx3D<> SIGMA = P*F.Transpose()/F.Det();
            Mtx3D<> S = F.Inv()*P;
            for(int k = 0; k < 9; ++k)
            {
                dat_sv[k] += 1./nquad * SIGMA[k];
                dat_sv[9+k] += 1./nquad * F[k];
                if(k < 6) dat_dsv[6*iQuad + k] = D_alpha_Sv[k];
            }
            dat_sv[18] += 1./nquad * S[0];
            dat_sv[19] += 1./nquad * (S[1]+S[3])/2;
            dat_sv[20] += 1./nquad * (S[2]+S[6])/2;
            dat_sv[21] += 1./nquad * S[4];
            dat_sv[22] += 1./nquad * (S[5]+S[7])/2;
            dat_sv[23] += 1./nquad * S[8];      
        }
        return Ani::TENSOR_GENERAL;
    };
    auto dP_tensor = [comp_gradU, comp_lagrange_coef, dP_func, W_params](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iQuad){
        (void) dims;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double p_coef = comp_lagrange_coef(X, *p.pXYZ, p.pdofs, *p.palloc);
        PotentialParams c = W_params(*p.c, X);
        int offset = 6*(p.SoE_coefs.nexp+2)*iQuad;
        auto dP = tensor_convert<Tensor4Rank<3>>(dP_func(c, grU, p_coef, p.svqdofs.begin()+offset, p.SoE_coefs));
        std::copy(dP.m_dat.data(), dP.m_dat.data() + 81, D);
        return Ani::TENSOR_GENERAL;
    };
    auto dlagr_tensor = [comp_gradU, dJ_func](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto P = dJ_func(grU);
        std::copy(P.m_dat.data(), P.m_dat.data() + 9, D);
        return Ani::TENSOR_GENERAL;
    };
    auto lagr_rhs_tensor = [comp_gradU, dJ_func](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        D[0] = (Mech::J<>(0, Mech::grU_to_E(grU))() - 1);
        return Ani::TENSOR_SCALAR;
    };
    auto P_fuse_tensor = [P_tensor,comp_gradU,nquad,tag_Sv](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem){
        auto& p = *static_cast<ProbLocData*>(user_data);
        real_array dat_sv = tag_Sv[*p.c];
        std::fill(dat_sv.begin(), dat_sv.end(), 0.0);
        for(std::size_t r = 0; r < mem.f; ++r) // over tets
        for(std::size_t n = 0; n < mem.q; ++n) // over quad points
        {
            DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
            P_tensor({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]},
                            Dloc.data, Ddims, user_data, n);
        }
        return Ani::TENSOR_GENERAL;
    };
    auto dP_fuse_tensor = [dP_tensor](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem){
        for(std::size_t r = 0; r < mem.f; ++r) // over tets
        for(std::size_t n = 0; n < mem.q; ++n) // over quad points
        {
            DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
            dP_tensor({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]},
                            Dloc.data, Ddims, user_data, n);
        }
        return Ani::TENSOR_GENERAL;
    };
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel, unf, pnf, nquad, &tag_SvQ, &SoE_coefs, um1, um2, up, quad_order, geom_mask = (UFem.dofMap().GetGeomMask() | DofT::FACE)](ElementalAssembler& p) -> void {
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.rho = ::rho;
        data.lbl.fillFromBndTag(BndLabel, geom_mask, *p.nodes, *p.edges, *p.faces);
        data.udofs.Init(p.vars->begin(0), unf);
        data.pdofs.Init(p.vars->begin(1), pnf);
        std::vector<double> mem1(unf), mem2(unf);
        data.um1dofs.Init(mem1.data(), unf);
        data.um2dofs.Init(mem2.data(), unf);

        std::vector<Tag> up1 = {um1,up[1]}, up2 = {um2,up[1]};
        ElementalAssembler::GatherDataOnElement(up1, p, data.um1dofs.data, 0);
        ElementalAssembler::GatherDataOnElement(up2, p, data.um2dofs.data, 0);
        data.c = p.cell;
        data.SoE_coefs = SoE_coefs;
        
        unsigned FuncDim = 6*(data.SoE_coefs.nexp+2);
        auto data_sz = FuncDim*nquad;
        auto data_chunk = p.pool->alloc(data_sz, 0, 0);
        data.svqdofs = ArrayView<>(data_chunk.m_mem.ddata, data_sz);
        // initialize allocated memory with tag data
        Storage::real_array dat_svq = tag_SvQ[*p.cell];
        copy_quadrature_data_from_proper_to_custom_order(dat_svq.data(), data.svqdofs.data, quad_order, p.m, p.nodes->data(), FuncDim);
        // setup history variables Q_k^n (only at first iteration)
        if(data.SoE_coefs.first_iter)
            //data.SoE_coefs.setup_step(dT, dat_svq.data(), nquad);
            data.SoE_coefs.setup_step(dT, data.svqdofs.data, nquad);
        p.compute(args, &data);
        // move computed data back into tag
        copy_quadrature_data_from_custom_to_proper_order(data.svqdofs.data, dat_svq.data(), quad_order, p.m, p.nodes->data(), FuncDim);
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_jacobian_assembler =
        [unf, pnf, mass_mtx_tensor, dP_fuse_tensor, dlagr_tensor, &UFem, &PFem, order = quad_order](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc) {
        (void) rw, (void) iw;
        DenseMatrix<> A(Adat, unf+pnf, unf+pnf); A.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc((unf+pnf)*(unf+pnf), 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf+pnf, unf+pnf); B.SetZero();
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto& d = *static_cast<ProbLocData*>(user_data);
        d.pXYZ = &XYZ, d.palloc = &adapt_alloc;
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN), iden_p = PFem.getOP(IDEN);
        // mass matrix: rho u_i / dt^2
        fem3Dtet<DfuncTraits<>>(XYZ, iden_u, iden_u, mass_mtx_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            for (std::size_t j = 0; j < unf; ++j)
                A(i, j) = B(i, j);
        // elemental stiffness matrix <dP grad(P2^3), grad(P2^3)>
        fem3Dtet<DfuncTraitsFusive<>>(XYZ, grad_u, grad_u, dP_fuse_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            for (std::size_t j = 0; j < unf; ++j)
                A(i, j) += B(i, j);
        // \int (dP(\nabla u)/dp p) : \nabla \phi
        // \int q d(J - 1)/dF : \nabla u
        fem3Dtet<DfuncTraits<>>(XYZ, iden_p, grad_u, dlagr_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            for (std::size_t j = 0; j < pnf; ++j)
                A(i, unf+j) = A(unf+j, i) = B(i, j);
        // choose boundary parts of the tetrahedron
        DofT::TetGeomSparsity sp = d.lbl.getSparsity(DIRICHLET_BND);
        if (!sp.empty())
            applyDirMatrix(*UFem.dofMap().target<>(), A, sp);
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_residual_assembler =
        [unf, pnf, mass_rhs_tensor, P_fuse_tensor, lagr_rhs_tensor, &UFem, &PFem, order = quad_order, grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN)](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc) {
        (void) rw, (void) iw;
        DenseMatrix<> F(Adat, unf+pnf, 1); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc(unf+pnf, 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf+pnf, 1); B.SetZero();
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto& d = *static_cast<ProbLocData*>(user_data);
        d.pXYZ = &XYZ, d.palloc = &adapt_alloc;
        auto iden_p = PFem.getOP(IDEN);
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        // mass rhs: rho u_tt^n
        fem3Dtet<DfuncTraits<>>(XYZ, iden_p0, iden_u, mass_rhs_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            F[i] = B[i];
        // elemental stiffness matrix <P, grad(P2^3)>
        fem3Dtet<DfuncTraitsFusive<>>(XYZ, iden_p0, grad_u, P_fuse_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            F[i] += B[i];
        // elemental right hand side vector <(J-1), P1>
        fem3Dtet<>(XYZ, iden_p0, iden_p, lagr_rhs_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < pnf; ++i)
            F[i+unf] = B[i];
        // no RHS here, please add if needed
        // choose boundary parts of the tetrahedron
        DofT::TetGeomSparsity sp = d.lbl.getSparsity(DIRICHLET_BND);
        if (!sp.empty())
            applyDirResidual(*UFem.dofMap().target<>(), F, sp);
    };
    //define assembler
    Assembler discr(m);
    discr.SetMatFunc(GenerateElemMat(local_jacobian_assembler, unf+pnf, unf+pnf, 0, 0));
    discr.SetRHSFunc(GenerateElemRhs(local_residual_assembler, unf+pnf, 0, 0));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*UFem.base()), Var1Helper = GenerateHelper(*PFem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        fed.PushTrialFunc(Var1Helper, "p");
        fed.PushTestFunc(Var1Helper, "phi_p");
        discr.SetProbDescr(std::move(fed));
    }
    discr.pullInitValFrom(up);
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();
    if (pRank == 0) std::cout << "#dofs = " << discr.m_enum.getMatrixSize() << std::endl;
    //get parallel interval and allocate parallel vectors
    auto i0 = discr.getBegInd(), i1 = discr.getEndInd();
    Sparse::Matrix  A( "A" , i0, i1, m->GetCommunicator());
    Sparse::Vector  x( "x" , i0, i1, m->GetCommunicator()),
                    dx("dx", i0, i1, m->GetCommunicator()),
                    b( "b" , i0, i1, m->GetCommunicator());
    //discr.AssembleTemplate(A);  //< preallocate memory for matrix (to accelerate matrix assembling), this call is optional
    // set options to use preallocated matrix state (to accelerate matrix assembling)
    Ani::AssmOpts opts = Ani::AssmOpts()/*.SetIsMtxIncludeTemplate(true)
                                        .SetUseOrderedInsert(true)
                                        .SetIsMtxSorted(true)*/; //< setting this parameters is optional
    auto num_dofs = UFem.dofMap().NumDofs();
    auto inum_dofs = DofTNumDofsToInmostNumDofs(num_dofs);
    std::vector<double> udat(unf);
    // Ani::DynMem<> dmem; //< можно и это использовать
    std::vector<char> mem;
    PlainMemoryX<> mem_req = UFem.interpolateByDOFs_mem_req();
    mem.resize(mem_req.enoughRawSize());
    mem_req.allocateFromRaw(mem.data(), mem.size());
    auto U_t = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int {
        (void) X; (void) dim; (void) user_data;
        auto r = displacement(X, T);
        std::copy(r.begin(), r.end(), res);
        return 0;
    };
    auto interpolate_on_element = [&mem_req, &UFem](auto U_t, Element e, double* udat) -> DofT::TetGeomSparsity {
        std::array<HandleType, 4> nds;
        std::array<HandleType, 6> eds;
        std::array<HandleType, 4> fcs;
        Cell c = e.getCells()[0];
        HandleType ch = c.GetHandle();
        Ani::collectConnectivityInfo(c, nds.data(), eds.data(), fcs.data(), true, true);
        std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
        const bool comp_node_perm = (UFem.dofMap().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
        if (comp_node_perm){ //< для твоих пространств comp_node_perm == false
            std::array<long, 4> gni;
            for (int i = 0; i < 4; ++i)
                gni[i] = Node(c.GetMeshLink(), nds[i]).GlobalID();
            canonical_node_indexes = createOrderPermutation(gni.data());
        }
        double P[12]{};
        for (int n = 0; n < 4; ++n)
            for (int k = 0; k < 3; ++k)
                P[3*canonical_node_indexes[n] + k] = Node(c.GetMeshLink(), nds[n]).Coords()[k];
        Ani::Tetra<const double> XYZ(P+0, P+3, P+6, P+9);
        HandleType* elems[4]{nds.data(), eds.data(), fcs.data(), &ch};
        std::size_t elems_sz[4]{4, 6, 4, 1};
        int ielem_type_num = ElementNum(e->GetElementType());
        DofT::TetGeomSparsity sp;
        for (std::size_t i = 0; i < elems_sz[ielem_type_num]; ++i)
            if (e->GetHandle() == elems[ielem_type_num][i])
                sp.set(ielem_type_num, i);
        // GatherDataOnElement(u0, *UFem.dofMap().target<>(), c.GetMeshLink(), c.GetHandle(), fcs.data(), eds.data(), nds.data(),
        //                     canonical_node_indexes.data(), udat.data(), nullptr, 0);
        UFem.interpolateByDOFs(XYZ, U_t, ArrayView<>(udat, UFem.dofMap().NumDofOnTet()), sp, mem_req /*dmem*/, nullptr);
        return sp;
    };
    auto update_dynamic_bc = [m,mask,bmrk,&udat,U_t,&UFem,&inum_dofs,&discr,&BndLabel,interpolate_on_element,umask](TagRealArray u0, Sparse::Vector& x){
        for (auto e = m->BeginElement(umask); e != m->EndElement(); ++e) if(e->HaveData(BndLabel) && e->Integer(BndLabel) & DIRICHLET_BND && e->GetStatus() != Element::Ghost){
            DofT::TetGeomSparsity sp = interpolate_on_element(U_t, e->getAsElement(), udat.data());
            for (auto it = UFem.dofMap().beginBySparsity(sp, true); it != UFem.dofMap().endBySparsity(); ++it){
                auto lo = *it;
                //auto lto = lo.getGeomOrder();
                // Element elem(c.GetMeshLink(), elems[DofT::GeomTypeDim(lo.etype)][lo.nelem]);
                Element elem = e->getAsElement();
                // код для сохранения данных в вектор
                auto elem_dofs_per_dim = inum_dofs[DofT::GeomTypeDim(lo.etype)] / 3; //3 - dimension of u
                IGlobEnumeration::NaturalIndex nid(elem, lo.etype, 0, lo.leid / elem_dofs_per_dim, lo.leid % elem_dofs_per_dim);
                auto vid = discr.m_enum(nid).id;
                x[vid] = udat[lo.gid];
                // код для сохранения данных в тег
                //elem->RealArray(u0)[lo.leid] = udat[lo.gid];
            }
        }
    };
    //setup linear solver
    Solver lin_solver(p.lin_sol_nm, p.lin_sol_prefix);
    auto assemble_R = [&discr, &up](const Sparse::Vector& x, Sparse::Vector &b) -> int {
        discr.SaveSolution(x, up);
        std::fill(b.Begin(), b.End(), 0.0);
        return discr.AssembleRHS(b);
    };
    auto assemble_J = [&discr, &up, opts](const Sparse::Vector& x, Sparse::Matrix &A) -> int {
        discr.SaveSolution(x, up);
        std::for_each(A.Begin(), A.End(), [](INMOST::Sparse::Row& row){ for (auto vit = row.Begin(); vit != row.End(); ++vit) vit->second = 0.0; });
        return discr.AssembleMatrix(A, opts);
    };
    auto vec_norm = [m](const Sparse::Vector& x)->double {
        double lsum = 0, gsum = 0;
        for (auto itx = x.Begin(); itx != x.End(); ++itx)
            lsum += (*itx) * (*itx);
        gsum = m->Integrate(lsum);
        return sqrt(gsum);
    };
    auto vec_saxpy = [](double a, const Sparse::Vector& xv, double b, const Sparse::Vector& yv, Sparse::Vector& zv) {
        auto itz = zv.Begin();
        for (auto itx = xv.Begin(), ity = yv.Begin();
            itx != xv.End() && ity != yv.End() && itz != zv.End(); ++itx, ++ity, ++itz)
            *itz = a * (*itx) + b * (*ity);
    };
    auto compute_gradU_at_point = [gradUFEM = UFem.getOP(GRAD)](const Cell& c, const Coord<> &X, Assembler& discr, Tag u)->Mtx3D<> {
        return Mtx3D<>{eval_op_var_at_point<9>(c, X, gradUFEM, discr, u, std::initializer_list<int>{0}), true};
    };
    //setup nonlinear solver
    std::unique_ptr<SUNNonlinearSolver> nonlin_solver = std::make_unique<SUNNonlinearSolver>(lin_solver, A, x);
    {
        nonlin_solver->SetInfoHandlerFn([pRank](const char *module, const char *function, char *msg){
            if (pRank == 0) std::cout << "[" << module << "] " << function << "\n   " << msg << std::endl;
        });
        nonlin_solver->GetLinearSolverContent()->verbosity = 0;
        nonlin_solver->SetVerbosityLevel(2);
        nonlin_solver->SetAssemblerRHS(assemble_R).SetAssemblerMAT(assemble_J);
        double max_step = (2000*sqrt(m->Integrate(x.Size())) + 1);
        nonlin_solver->SetParameterReal("MaxNewtonStep", max_step);
        nonlin_solver->SetParameterIntegral("MaxIters", p.nlin_maxit);
        //nonlin_solver->SetParameterReal("ScaledSteptol", 1e-7);
        nonlin_solver->SetParameterReal("FuncNormTol", p.nlin_abs_err);
        nonlin_solver->Init();
    }
    auto printNLSolverStatus = [&nonlin_solver,pRank,pCount](bool slvFlag, std::string prob_nm = "", bool exit_on_false = true) {
        auto success_solve = slvFlag;
        auto rank = pRank, npc = pCount;
        SUNNonlinearSolver& s = *nonlin_solver;
        if(!success_solve){
            if (exit_on_false){
                std::cout << prob_nm << ":\n"
                        <<"\t#NonLinIts = "<<s.GetNumNolinSolvIters() << " Residual = "<< s.GetResidualNorm()
                        << "\n\t#linIts = " << s.GetNumLinIters() << " #funcEvals = " << s.GetNumLinFuncEvals() << " #jacEvals = " << s.GetNumJacEvals()
                        << "\n\t#convFails = " << s.GetNumLinConvFails() << " #betaCondFails = " << s.GetNumBetaCondFails() << " #backtrackOps = " << s.GetNumBacktrackOps() << "\n";
                std::cout<<"\t"<<rank<< " / " << npc << " failed to solve system. ";
                std::cout << "Reason: " << s.GetReason() << std::endl;
                exit(-1);
            } else {
            if (rank == 0)
                    std::cout << prob_nm << ":\n"
                        <<"\tnot_converged #NonLinIts = "<<s.GetNumNolinSolvIters() << " Residual = "<< s.GetResidualNorm()
                        << "\n\t#linIts = " << s.GetNumLinIters() << " #funcEvals = " << s.GetNumLinFuncEvals() << " #jacEvals = " << s.GetNumJacEvals()
                        << "\n\t#convFails = " << s.GetNumLinConvFails() << " #betaCondFails = " << s.GetNumBetaCondFails() << " #backtrackOps = " << s.GetNumBacktrackOps() << "\n"
                        << "\n\tMatAssmTime = " << s.GetMatAssembleTime() << "s RHSAssmTime = " << s.GetRHSAssembleTime() << "s"
                        << "Reason: " << s.GetReason() << std::endl;
            }
        }
        else{
            if(rank == 0)
                std::cout << prob_nm << ":\n"
                        <<"\tsolved_succesful #NonLinIts = "<<s.GetNumNolinSolvIters() << " Residual = "<< s.GetResidualNorm()
                        << "\n\t#linIts = " << s.GetNumLinIters() << " #funcEvals = " << s.GetNumLinFuncEvals() << " #jacEvals = " << s.GetNumJacEvals()
                        << "\n\t#convFails = " << s.GetNumLinConvFails() << " #betaCondFails = " << s.GetNumBetaCondFails() << " #backtrackOps = " << s.GetNumBacktrackOps()
                        << "\n\tMatAssmTime = " << s.GetMatAssembleTime() << "s RHSAssmTime = " << s.GetRHSAssembleTime() << "s" << std::endl;
        }
    };
    int step = 0;
    T = 0;
    TagRealArray tag_gU = m->CreateTag("gradU", DATA_REAL, CELL, NONE, 9+3);
    m->SetFileOption("Tag:" + tag_SvQ.GetTagName(), "nosave");
    std::ofstream ofs(p.save_dir+"viscoelastic1.csv");
    ofs << "T;ang_displ;torque;traction" << std::endl;
    while(T < T_FINAL-1.0e-20)
    {
        double torque_direct = 0.0, traction_direct = 0.0;
        T += dT;
        ++step;
        update_dynamic_bc(u, x);
        TimerWrap m_timer_total; m_timer_total.reset();
        SoE_coefs.first_iter = 1;
        assemble_R(x, b);
        SoE_coefs.first_iter = 0;
        double anrm = vec_norm(b), rnrm = 1;
        double anrm0 = anrm;
        int ni = 0;
        if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        if(use_kinsol) {
            bool slvFlag = nonlin_solver->Solve(SUNNonlinearSolver::LINESEARCH);
            printNLSolverStatus(slvFlag, "nonlin_viscoelastic_incompress");
        }
        else {
            while (rnrm >= p.nlin_rel_err && anrm >= p.nlin_abs_err && ni < p.nlin_maxit) {
                assemble_J(x, A);
                if (pRank == 0) std::cout << "--- Compute preconditioner ---" << std::endl;
                lin_solver.SetMatrix(A);
                if (pRank == 0) std::cout << "--- Solve linear system ---" << std::endl;
                if (std::stod(lin_solver.GetParameter("absolute_tolerance")) > p.lin_abs_scale*anrm)
                    lin_solver.SetParameterReal("absolute_tolerance", p.lin_abs_scale*anrm);
                lin_solver.Solve(b, dx);
                print_linear_solver_status(lin_solver, prob_name, true);
                vec_saxpy(1, x, -1, dx, x);
                assemble_R(x, b);
                anrm = vec_norm(b);
                rnrm = anrm / anrm0;
                ni++;
                //m->Save(p.save_dir + p.save_prefix + std::to_string(step) + "_iter" + std::to_string(ni) + ".pvtu");
                if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
            }
        }
        for (auto e = m->BeginElement(umask); e != m->EndElement(); ++e) {
            auto arr = e->RealArray(u), arrm1 = e->RealArray(um1), arrm2 = e->RealArray(um2);
            std::copy(arrm1.begin(), arrm1.end(), arrm2.begin());
            std::copy(arr.begin(), arr.end(), arrm1.begin());
        }
        bool once = true;
        for(int k = 0; k < m->CellLastLocalID(); ++k) if(m->isValidCell(k))
        {
            Cell c = m->CellByLocalID(k);
            Storage::real_array dat_gU = tag_gU[c];
            std::array<double,3> cnt; c.Centroid(cnt.data());
            Mtx3D<> gU = compute_gradU_at_point(c, cnt, discr, u);
            std::copy(gU.begin(), gU.begin()+9, dat_gU.begin());

            if(c.nbAdjElements(FACE, bmrk))
            {
                ElementArray<Element> bndfaces = c.getAdjElements(FACE, bmrk);
                Face fbnd = InvalidFace();
                for(uint q = 0; q < bndfaces.size() && !fbnd.isValid(); ++q)
                    if(bndfaces[q].Integer(BndLabel) & DIRICHLET_BND_TOP)
                        fbnd = bndfaces[q].getAsFace();
                if(fbnd.isValid())
                {
                    // find deformed area: get points (reference configuration), move them into current configuration, compute triangle area
                    ElementArray<Node> fnodes = fbnd.getNodes();
                    Coord<> xref[3], xcur[3];
                    for(int q = 0; q < 3; ++q)
                    {
                        fnodes[q].Centroid(xref[q].data());
                        for(int r = 0; r < 3; ++r)
                            xcur[q][r] = position(xref[q],T,r);
                    }
                    double farea = tri_area(xcur[0].data(), xcur[1].data(), xcur[2].data());
                    Storage::real_array dat_sv = tag_Sv[c];
                    double dtorq, dtrac;
                    std::array<double,3> fcnt; fbnd.Centroid(fcnt.data());
                    dtorq = position(fcnt,T,0)*(dat_sv[5]+dat_sv[7])/2 - position(fcnt,T,1)*(dat_sv[2]+dat_sv[6])/2;
                    dtrac = dat_sv[8];
                    torque_direct += dtorq*farea;
                    traction_direct += dtrac*farea;
                    if(once)
                    {
                        std::cout << "T " << T << " area: reference " << fbnd.Area() << " current " << farea
                            << " Aref/Acur " << (fbnd.Area() / farea) << std::endl;
                        once = false;
                    }
                }
            }
            
            dat_gU[9] = dat_gU[10] = dat_gU[11] = 0.0;
            ElementArray<Edge> cedges = c.getEdges();
            for(unsigned q = 0; q < cedges.size(); ++q)
            {
                Storage::real_array dat_eu = cedges[q].RealArray(u);
                dat_gU[9]  += 1./cedges.size() * dat_eu[0];
                dat_gU[10] += 1./cedges.size() * dat_eu[1];
                dat_gU[11] += 1./cedges.size() * dat_eu[2];
            }
        }
        torque_direct = m->Integrate(torque_direct);
        traction_direct = m->Integrate(traction_direct);
        ofs << T << ";" << psi(T)
            << ";" << torque_direct << ";" << traction_direct << std::endl;
        double total_sol_time =  m_timer_total.elapsed();
        if (pRank == 0) std::cout << "Step solution time: " << total_sol_time << "s" << std::endl;
        if (step % save_steps == 0)
        {
            std::string save_name = p.save_dir + p.save_prefix + std::to_string(step) + ".pvtu";
            m->Save(save_name);
            if (pRank == 0) std::cout << "[T=  " << T << "] Intermediate result saved in \"" << save_name << "\"" << std::endl;
        }
    }
    m->ReleaseMarker(bmrk, FACE);
    ofs.close();
    discr.Clear();
    delete m;
    InmostFinalize();
    return 0;
}
