#ifndef _SOE_HPP
#define _SOE_HPP

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "inmost.h"
#include "anifem++/autodiff/cauchy_strain_invariants.h"

template<typename T>
T from_string(std::string s)
{
        T val;
        std::stringstream ss(s);
        ss >> val;
        return val;
}

using namespace Ani;
typedef INMOST::Storage::real_array real_array;
// (0 1 2)
// (1 3 4)
// (2 4 5)
typedef std::array<double,6> symmat3;//xx xy xz yy yz zz
// sum of exponents (soe)
struct soe_data
{
        unsigned nexp;
        std::vector<double> a;// b_k
        std::vector<double> b;// 1/tau_k
        double gamma = 0.0;   // b_0
        double alpha;
        int first_iter = 0;

        soe_data(unsigned nexp = 10, double sigma = 1.0) : nexp(nexp), a(nexp), b(nexp) {}

	double get_coef(double dt) const
	{
		double coef = gamma/dt;
		for(unsigned i = 0; i < nexp; ++i) coef += exp(-b[i]/2*dt)*a[i];
		return coef;
        }
        // Qh <-- (-gamm)*Sv^(n-1) + e^2_k Q^(n-1)_k
        // (D_alpha Sv)^n = gamm*Sv^n + Qh = b_0/dt (Sv^n - Sv^(n-1)) + sum_k Q_k^n
	/*SymMtx3D<> get_hist(double dt, const ArrayView<>& SvQ) const
	{
                SymMtx3D<> Qh;
                double gamm = get_coef(dt);
		for(int k = 0; k < 6; ++k) Qh[k] = -gamm*SvQ[k];
		for(int j = 0; j < nexp; ++j)
		{
			double mult = exp(-b[j]/2*dt);
			const double* Qj = &SvQ[6*(j+2)];
			for(int k = 0; k < 6; ++k)
				Qh[k] += mult*mult*Qj[k];
		}
	}*/
        void setup_step(double dt, double* SvQ, unsigned nquad)
        {
                // time step n-1 --> n
                // svq: Sv1(svq[0+j]), Sv0(svq[6+j]), Qk(svq[12+6*iQuad+j]) [j=0..5]
                // after step n-1 SvQ contains:
                //      Sv1=Sv^{n-1} , Sv0=Sv^{n-2} , Qk=Q_k^{n-1}
                // initialization for step n (iteration q=0):
                //   Sv^{n,0} = Sv^{n-1}
                //   Q_k^{n,0} = (e_k)^2 Q_k^{n-1} + e_k b_k (Sv^{n,0}-Sv^{n-1})
                //             = (e_k)^2 Q_k^{n-1}
                // after initialization (iteration q=0) for step n
                //                svq should contain:
                //      Sv1=Sv^{n,0} , Sv0=Sv^{n-1} , Qk=Q_k^{n,0}
                //      Sv1=Sv^{n-1} , Sv0=Sv^{n-1} , Qk=(e_k)^2 Q_k^{n-1}
                // operations:
                //      Qk  <-- (e_k)^2 Qk
                //      Sv0 <-- Sv1
                if(first_iter)
                {
                        for(unsigned iQuad = 0; iQuad < nquad; ++iQuad)
                        {
                                unsigned offset = 6*(nexp+2)*iQuad;
                                double* svq = SvQ + offset;
                                double *Sv1 = &svq[0], *Sv0 = &svq[6];
                                for(unsigned i = 0; i < nexp; ++i)
                                {
                                        double ek = exp(-b[i]/2*dt); //e_k = exp(-dt/2 * 1/tau_k))
                                        double* Qk = &svq[6*(i+2)];
                                        // Q_k^{n,0} = (e_k)^2 Q_k^(n-1)
                                        for(int j = 0; j < 6; ++j)
                                                Qk[j] *= ek*ek;
                                                //Qj[k] = mult*mult*Qj[k] + mult*a[j]*(svq[k]-svq[6+k]);
                                }
                                // Sv0 <-- Sv1
                                for(int j = 0; j < 6; ++j) Sv0[j] = Sv1[j];
                        }
                }
        }

        void update_iter(double dt, const SymMtx3D<>& Svnew, double* svq) const
        {
                // svq: Sv1(svq[0+j]) , Sv0(svq[6+j]), Qk(svq[12+6*iQuad+j]) [j=0..5]
                // time step n-1 --> n, iteration q --> q+1
                //     Svnew: Sv^{n,q+1}
                // after iteration q svq contains:
                // svq: Sv1=Sv^{n,q}  , Sv0=Sv^{n-1} , Qk=Q_k^{n,q}
                //     Q_k^{n,q}   = (e_k)^2 Q_k^(n-1) + e_k b_k (Sv^{n,q} - Sv^{n-1})
                // after iteration q+1 svq should contain:
                // svq: Sv1=Sv^{n,q+1}, Sv0=Sv^{n-1} , Qk=Q_k^{n,q+1}
                //     Q_k^{n,q+1} = (e_k)^2 Q_k^{n-1} + e_k b_k (Sv^{n,q+1} - Sv^{n-1})
                //
                // Q_k^{n,q+1} =   Q_k^{n,q}         + e_k b_k (Sv^{n,q+1} - Sv^{n,q})
                // Qk          <-- Qk                + e_k b_k (Svnew      - Sv1     )
                //
                // Q_k^{n,q+1} =   (e_k)^2 Q_k^(n-1) + e_k b_k (Sv^{n,q} - Sv^{n-1})
                //                                   + e_k b_k (Sv^{n,q+1} - Sv^{n,q})
                // Q_k^{n,q+1} =   (e_k)^2 Q_k^{n-1} + e_k b_k (Sv^{n,q+1} - Sv^{n-1})
                // operations:
                // Qk <-- Qk + e_k b_k (Svnew - S1) 
                // Sv1 <-- Svnew
		double *Sv1=&svq[0];
                for(unsigned i = 0; i < nexp; ++i)
		{
			double ek = exp(-b[i]/2*dt);
			double* Qk = &svq[6*(i+2)];
			for(int j = 0; j < 6; ++j)
				Qk[j] += ek*a[i]*(Svnew[j] - Sv1[j]);
		}
                // Sv1 <-- Svnew
                for(int j = 0; j < 6; ++j) Sv1[j] = Svnew[j];
        }

        SymMtx3D<> get_deriv(double dt, const SymMtx3D<>& Sv, double* svq) const
        {
                // svq: Sv1(svq[0+j]) , Sv0(svq[6+j]), Qk(svq[12+6*iQuad+j]) [j=0..5]
                // time step n-1 --> n, iteration q --> q+1
                //     Sv: Sv^{n,q+1}
                // 
                // svq: Sv1=Sv^{n,q}  , Sv0=Sv^{n-1} , Qk=Q_k^{n,q}
                update_iter(dt, Sv, svq); // q --> q+1
                // svq: Sv1=Sv^{n,q+1}, Sv0=Sv^{n-1} , Qk=Q_k^{n,q+1}
                // 
                // (D_alpha Sv)^{n,q+1} = b_0/dt (Sv^{n,q+1} - Sv^{n-1}) + sum_k Q_k^{n,q+1}
                // D_alpja <-- gamma/dt*(Sv1-Sv0) + sum_k Qk
                SymMtx3D<> D_alpha;
                double *Sv1 = &svq[0], *Sv0 = &svq[6];
                for(int j = 0; j < 6; ++j) D_alpha[j] = gamma/dt*(Sv1[j] - Sv0[j]);
		for(unsigned i = 0; i < nexp; ++i)
		{
			double* Qk = &svq[6*(i+2)];
			for(int j = 0; j < 6; ++j)
				D_alpha[j] += Qk[j];
		}       
                return D_alpha;       
        }
};
std::ostream& operator<<(std::ostream& os, const soe_data& icoef)
{
        os << icoef.alpha << " " << icoef.gamma;
        os << '\n';
        for(unsigned k = 0; k < icoef.nexp; ++k)
                os << " " << icoef.a[k];
        os << '\n';
	for(unsigned k = 0; k < icoef.nexp; ++k)
                os << " " << icoef.b[k];
        return os;
}
void read_soe_coefs(std::string filename, std::vector<soe_data>& coefs)
{
        std::ifstream ifs(filename);
        std::string s;
        unsigned nexp;
        ifs >> nexp;
        std::cout << "Read interpolation coefficients in csv format, " << (2*nexp+2) << " coefficients in each string." << std::endl;
        std::vector<double> vals;
        while(ifs.good())
        {
                vals.clear();
                ifs >> s;
                std::string::size_type p0 = 0, p = s.find_first_of(';');
                while(p != std::string::npos)
                {
                        vals.push_back(from_string<double>(s.substr(p0, p-p0)));
                        p0 = p+1;
                        p = s.find_first_of(';', p0);
                }
                vals.push_back(from_string<double>(s.substr(p0)));
                assert(vals.size() == 2*nexp+2);
                soe_data icoef = soe_data(nexp);
                icoef.alpha = vals[0];
                icoef.gamma = vals[1];
                for(unsigned k = 0; k < nexp; ++k)
                {
                        icoef.a[k] = vals[2+k];
                        icoef.b[k] = vals[2+k+nexp];
                }
                coefs.push_back(icoef);
        }
        ifs.close();
}
soe_data read_soe_coefs(std::string coefs_filename, double alpha, bool debug = false)
{
	std::vector<soe_data> icoefs;
        read_soe_coefs(coefs_filename, icoefs);
	soe_data acoefs;
        for(unsigned k = 0; k < icoefs.size(); ++k)
                if(fabs(icoefs[k].alpha - alpha) < 1.0e-6)
                {
                        acoefs = icoefs[k];
                        break;
                }
        if(debug)
        {
                acoefs.nexp = 0;
                acoefs.gamma = 1.0;
                acoefs.alpha = 1.0;
                acoefs.a.resize(0);
                acoefs.b.resize(0);
        }
	std::cout << "found SoE coefs: \n" << acoefs
                << " (debug " << debug << ")" << std::endl;
	return acoefs;
}

#endif //_SOE_HPP
