/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Volumetric path tracer", "Average path length", EAverage);

/*!\plugin[volpathsimple]{volpath\_simple}{Simple volumetric path tracer}
 * \order{3}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See
 *        page~\pageref{sec:strictnormals} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This plugin provides a basic volumetric path tracer that can be used to
 * compute approximate solutions of the radiative transfer equation. This
 * particular integrator is named ``simple'' because it does not make use of
 * multiple importance sampling. This results in a potentially
 * faster execution time. On the other hand, it also means that this
 * plugin will likely not perform well when given a scene that contains
 * highly glossy materials. In this case, please use \pluginref{volpath}
 * or one of the bidirectional techniques.
 *
 * This integrator has special support for \emph{index-matched} transmission
 * events (i.e. surface scattering events that do not change the direction
 * of light). As a consequence, participating media enclosed by a stencil shape (see
 * \secref{shapes} for details) are rendered considerably more efficiently when this
 * shape has \emph{no}\footnote{this is what signals to Mitsuba that the boundary is
 * index-matched and does not interact with light in any way. Alternatively,
 * the \pluginref{mask} and \pluginref{thindielectric} BSDF can be used to specify
 * index-matched boundaries that involve some amount of interaction.} BSDF assigned
 * to it (as compared to, say, a \pluginref{dielectric} or \pluginref{roughdielectric} BSDF).
 *
 * \remarks{
 *    \item This integrator performs poorly when rendering
 *      participating media that have a different index of refraction compared
 *      to the surrounding medium.
 *    \item This integrator has difficulties rendering
 *      scenes that contain relatively glossy materials (\pluginref{volpath} is preferable in this case).
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */
class SimpleVolumetricPathTracer : public MonteCarloIntegrator {
public:
	SimpleVolumetricPathTracer(const Properties &props) : MonteCarloIntegrator(props) { }

	/// Unserialize from a binary data stream
	SimpleVolumetricPathTracer(Stream *stream, InstanceManager *manager)
	 : MonteCarloIntegrator(stream, manager) { }

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec, std::vector<Spectrum> &densityDerivative, bool print_out) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		MediumSamplingRecord mRec;
		RayDifferential ray(r);
		Spectrum Li(0.0f);
		bool nullChain = true, scattered = false;
		Float eta = 1.0f;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		rRec.rayIntersect(ray);
		Spectrum throughput(1.0f);

		if (m_maxDepth == 1)
			rRec.type &= RadianceQueryRecord::EEmittedRadiance;

		/**
		 * Note: the logic regarding maximum path depth may appear a bit
		 * strange. This is necessary to get this integrator's output to
		 * exactly match the output of other integrators under all settings
		 * of this parameter.
		 */

		/* debug */
		bool DEBUG_TAMAR = print_out;
		if (DEBUG_TAMAR)
			cout << "start:" << endl;

//		Spectrum throuTmp = Spectrum(-1.0); //T!
//		Spectrum LiRRTmp = Spectrum(0.0);//T!
//		bool RR_f = false; //T!
		Float q = 1.0;

		while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
			/* ==================================================================== */
			/*                 Radiative Transfer Equation sampling                 */
			/* ==================================================================== */
			if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler, print_out)) {
				/* Sample the integral
				   \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
				*/

//				if (throuTmp[0] > 0.0){ //T!
//					float diffThrou = (throuTmp[0] - throughput[0]) +(throuTmp[1] - throughput[1]) + (throuTmp[2] - throughput[2]); //T!
//					if (diffThrou > 1e-6) {//T!
//						cout << "difference in throughput = " << diffThrou << endl; //T!
//						cout << "throuTmp = " << throuTmp.toString() << " throughput = " << throughput.toString() << endl; //T!
//					}
//				} //T!

				if (DEBUG_TAMAR) {
					cout << ray.toString();
					cout << "throughput: " << throughput.toString() << endl;
					cout << "mRec: [" << endl;
					cout << " sigmaS = " <<  (mRec.sigmaS).toString() << endl;
					cout << " pdfSuccess = " << mRec.pdfSuccess  << endl;
					cout << " transmittance = " << (mRec.transmittance).toString() << endl;
					cout << endl;
				}

				const PhaseFunction *phase = rRec.medium->getPhaseFunction();

				throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				// build densityDerivative
				for(std::vector<int>::size_type i = 0; i != mRec.scoreIndxs.size(); i++) {
					densityDerivative[mRec.scoreIndxs[i]] += Spectrum(mRec.scoreVals[i] * q);
				}
				mRec.scoreIndxs.clear();
				mRec.scoreVals.clear();

				if ((print_out) and (DEBUG_TAMAR)) {
					for(std::vector<int>::size_type i = 0; i != densityDerivative.size(); i++) {
						cout << "densityDerivative at " << i << " = " << densityDerivative[i].toString() << endl;
					}
				}

				/* ==================================================================== */
				/*                     Direct illumination sampling                     */
				/* ==================================================================== */

				/* Estimate the single scattering component if this is requested */
				if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance) {
					// Tamar - goes into this section in multiple scattering as well
					DirectSamplingRecord dRec(mRec.p, mRec.time);
					int maxInteractions = m_maxDepth - rRec.depth - 1;

					Spectrum value = scene->sampleAttenuatedEmitterDirect(
							dRec, rRec.medium, maxInteractions,
							rRec.nextSample2D(), rRec.sampler);

					if (!value.isZero()) {
						Float phaseVal = phase->eval(
								PhaseFunctionSamplingRecord(mRec, -ray.d, dRec.d));
						Li += throughput * value * phaseVal;
						MediumSamplingRecord mdRec;
						RayDifferential rayD = RayDifferential(mRec.p, dRec.d ,mRec.time); //18_3

						rRec.medium->derivateDensity(rayD, mdRec, true, print_out, dRec.dist);//18_3

						for(std::vector<int>::size_type i = 0; i != mdRec.scoreIndxs.size(); i++) {
							if (print_out) {
								cout << "LE:" << endl;
								cout << densityDerivative[mRec.scoreIndxs[i]].toString() << endl;
							}

							densityDerivative[mdRec.scoreIndxs[i]] += Spectrum(mdRec.scoreVals[i] * phaseVal * q); // * value 
							if (print_out)
								cout << densityDerivative[mRec.scoreIndxs[i]].toString() << endl;
						}
						mdRec.scoreIndxs.clear();
						mdRec.scoreVals.clear();
					}
				}

				/* Stop if multiple scattering was not requested, or if the path gets too long */
				if ((rRec.depth + 1 >= m_maxDepth && m_maxDepth > 0) ||
					!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
					break;

				/* ==================================================================== */
				/*             Phase function sampling / Multiple scattering            */
				/* ==================================================================== */

				PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
				Float phaseVal = phase->sample(pRec, rRec.sampler);
				if (phaseVal == 0)
					break;
				throughput *= phaseVal;
				q *= phaseVal; //18_3

				/* Trace a ray in this direction */
				ray = Ray(mRec.p, pRec.wo, ray.time);
				ray.mint = 0;
				scene->rayIntersect(ray, its);
				nullChain = false;
				scattered = true;

			} else {
				/* Sample
					tau(x, y) * (Surface integral). This happens with probability mRec.pdfFailure
					Account for this and multiply by the proper per-color-channel transmittance.
				*/

//				Spectrum LiTmp = Li; //T!
//				throuTmp = throughput; //T!

				if (rRec.medium) {
					throughput *= mRec.transmittance / mRec.pdfFailure;
//					Spectrum factor = mRec.transmittance / mRec.pdfFailure; //T!
//					if ((factor[0]!= 1.0) and (factor[1]!=1.0) and (factor[2]!=1.0))
//						cout << "factor = " <<  factor.toString() << endl;
				}

				if (!its.isValid()) {
					/* If no intersection could be found, possibly return
					   attenuated radiance from a background luminaire */
					if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
						&& (!m_hideEmitters || scattered)) {
						Spectrum value = throughput * scene->evalEnvironment(ray);
						if (rRec.medium) {
//							Spectrum val_diff = rRec.medium->evalTransmittance(ray);
//							value *= val_diff;
							value *= rRec.medium->evalTransmittance(ray);
//							if ((val_diff[0] != 1.0) or (val_diff[1] != 1.0) or (val_diff[2] != 1.0)) {
//								cout << "rRec.medium " << " 0 = " << val_diff[0] << " 1 = " << val_diff[1] << " 2 = " << val_diff[2] << endl;
//							}
							if (DEBUG_TAMAR) {
								cout << "throughput: " << throughput.toString() << endl;
								cout << "mRec: [" << endl;
								cout << " pdfFailure = " << mRec.pdfFailure  << endl;
								cout << " transmittance = " << (mRec.transmittance).toString() << endl;
								cout << "value: " << value.toString() << endl;
								cout << endl;
							}
						}
						if (!value.isZero()){
							cout << "else, its.is valid == 0 and rRec type is RadianceQueryRecord::EEmittedRadiance = " << rRec.type << " scattered is " << scattered << " m_hideEmitters is " << m_hideEmitters << endl;
							cout << "valie is " << value.toString() << endl;
						}
						Li += value;
//						float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//						if (diffLi > 1e-6) { //T!
//							cout << "diff = " << diffLi << endl;
//							cout << "If no intersection could be found, possibly return attenuated radiance from a background luminaire" << endl; //T!
//							cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//							cout << "Li = " << Li.toString() << endl; //T!
//						} //T!
					}
//					float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//					if (diffLi > 1e-6) { //T!
//						cout << "If no intersection could be found" << endl; //T!
//						cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//						cout << "Li = " << Li.toString() << endl; //T!
//					} //T!
					break;
				}

				/* Possibly include emitted radiance if requested */
				if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
					&& (!m_hideEmitters || scattered)){
					cout << "else, its.isEmitter() is " << its.isEmitter() << " and rRec.type is RadianceQueryRecord::EEmittedRadiance = " << rRec.type << endl;
					Li += throughput * its.Le(-ray.d);
//					float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//					if (diffLi > 1e-6) { //T!
//						cout << "Possibly include emitted radiance if requested" << endl; //T!
//						cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//						cout << "Li = " << Li.toString() << endl; //T!
//					} //T!
				}

				/* Include radiance from a subsurface integrator if requested */
				if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)){
					cout << "else, its.isEmitter() is " << its.hasSubsurface() << " and rRec.type is RadianceQueryRecord::ESubsurfaceRadiance = " << rRec.type << endl;
					Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);
//					float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//					if (diffLi > 1e-6) { //T!
//						cout << "Include radiance from a subsurface integrator if requested" << endl; //T!
//						cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//						cout << "Li = " << Li.toString() << endl; //T!
//					} //T!
				}

				/* Prevent light leaks due to the use of shading normals */
				Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
					  wiDotShN  = Frame::cosTheta(its.wi);
				if (m_strictNormals && wiDotGeoN * wiDotShN < 0){
//					float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//					if (diffLi > 1e-6) { //T!
//						cout << "1 - Prevent light leaks due to the use of shading normals" << endl; //T!
//						cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//						cout << "Li = " << Li.toString() << endl; //T!
//					} //T!
					break;
				}
				/* ==================================================================== */
				/*                     Direct illumination sampling                     */
				/* ==================================================================== */

				const BSDF *bsdf = its.getBSDF(ray);

				/* Estimate the direct illumination if this is requested */
				if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
						(bsdf->getType() & BSDF::ESmooth)) {
					DirectSamplingRecord dRec(its);
					int maxInteractions = m_maxDepth - rRec.depth - 1;

					Spectrum value = scene->sampleAttenuatedEmitterDirect(
							dRec, its, rRec.medium, maxInteractions,
							rRec.nextSample2D(), rRec.sampler);

					if (!value.isZero()) {
						/* Allocate a record for querying the BSDF */
						BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));
						bRec.sampler = rRec.sampler;

						Float woDotGeoN = dot(its.geoFrame.n, dRec.d);
						/* Prevent light leaks due to the use of shading normals */
						if (!m_strictNormals ||
							woDotGeoN * Frame::cosTheta(bRec.wo) > 0) {
							cout << "else, BSDF m_strictNormals is zero = " << m_strictNormals << " and woDotGeoN * Frame::cosTheta(bRec.wo) > 0" << endl;
							Li += throughput * value * bsdf->eval(bRec);
//							float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//							if (diffLi > 1e-6) { //T!
//								cout << "Direct illumination - Prevent light leaks due to the use of shading normals" << endl; //T!
//								cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//								cout << "Li = " << Li.toString() << endl; //T!
//							} //T!
						}
					}
				}

				/* ==================================================================== */
				/*                   BSDF sampling / Multiple scattering                */
				/* ==================================================================== */

				/* Sample BSDF * cos(theta) */
				BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
				Spectrum bsdfVal = bsdf->sample(bRec, rRec.nextSample2D());
				if (bsdfVal.isZero()){
//					cout << "bsdfVal = 0" << endl;
					break;
				}
				if ((bsdfVal[0] != 1.0) && (bsdfVal[1] != 1.0) && (bsdfVal[2] != 1.0)){
					cout << "bsdfVal = " << bsdfVal.toString() << endl;
				}
				/* Recursively gather indirect illumination? */
				int recursiveType = 0;
				if ((rRec.depth + 1 < m_maxDepth || m_maxDepth < 0) &&
					(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)) {
//					float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//					if (diffLi > 1e-6) { //T!
//						cout << "diff = " << diffLi << endl;
//						cout << "Recursively gather indirect illumination" << endl; //T!
//						cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//						cout << "Li = " << Li.toString() << endl; //T!
//					} //T!
					recursiveType |= RadianceQueryRecord::ERadianceNoEmission;
				}
				/* Recursively gather direct illumination? This is a bit more
				   complicated by the fact that this integrator can create connection
				   through index-matched medium transitions (ENull scattering events) */
				if ((rRec.depth < m_maxDepth || m_maxDepth < 0) &&
					(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
					(bRec.sampledType & BSDF::EDelta) &&
					(!(bRec.sampledType & BSDF::ENull) || nullChain)) {
					recursiveType |= RadianceQueryRecord::EEmittedRadiance;
					nullChain = true;
//					float diffLi = (LiTmp[0] - Li[0])+(LiTmp[1] - Li[1])+(LiTmp[2] - Li[2]); //T!
//					if (diffLi > 1e-6) { //T!
//						cout << "Recursively gather direct illumination" << endl; //T!
//						cout << "LiTmp = " << LiTmp.toString() << endl; //T!
//						cout << "Li = " << Li.toString() << endl; //T!
//					} //T!
				} else {
					nullChain &= bRec.sampledType == BSDF::ENull;
				}

				/* Potentially stop the recursion if there is nothing more to do */
				if (recursiveType == 0)
					break;
				rRec.type = recursiveType;

				/* Prevent light leaks due to the use of shading normals */
				const Vector wo = its.toWorld(bRec.wo);
				Float woDotGeoN = dot(its.geoFrame.n, wo);
				if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
					break;

				/* Keep track of the throughput, medium, and relative
				   refractive index along the path */
				throughput *= bsdfVal;
				eta *= bRec.eta;
				if (its.isMediumTransition())
					rRec.medium = its.getTargetMedium(wo);

				/* In the next iteration, trace a ray in this direction */
				ray = Ray(its.p, wo, ray.time);
				scene->rayIntersect(ray, its);
				scattered |= bRec.sampledType != BSDF::ENull;
			}

			if (rRec.depth++ >= m_rrDepth) {
				//Tamar - for now depth is always inf
				/* Russian roulette: try to keep path weights equal to one,
				   while accounting for the solid angle compression at refractive
				   index boundaries. Stop with at least some probability to avoid
				   getting stuck (e.g. due to total internal reflection) */
//				LiRRTmp = Li; //T!
				Float q_tmp = std::min(throughput.max() * eta * eta, (Float) 0.95f);
				if (rRec.nextSample1D() >= q_tmp)
					break;
				q /= q_tmp;
				throughput /= q_tmp;
//				RR_f = true; //T!
			}
		}
//		if (RR_f == true) {
//			float diffLi = (LiRRTmp[0] - Li[0])+(LiRRTmp[1] - Li[1])+(LiRRTmp[2] - Li[2]); //T!
//			if (diffLi > 1e-6) { //T!
//				cout << "RR" << endl; //T!
//				cout << "LiRRTmp = " << LiRRTmp.toString() << endl; //T!
//				cout << "Li = " << Li.toString() << endl; //T!
//			} //T!
//		}
		avgPathLength.incrementBase();
		avgPathLength += rRec.depth;
		return Li;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		MonteCarloIntegrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "SimpleVolumetricPathTracer[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(SimpleVolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SimpleVolumetricPathTracer, "Simple volumetric path tracer");
MTS_NAMESPACE_END
