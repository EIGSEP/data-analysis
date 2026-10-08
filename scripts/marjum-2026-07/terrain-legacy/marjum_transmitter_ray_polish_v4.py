"""Joint polish in picked-ray/range coordinates with strict horizon guards."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from marjum_camera import rays,project,radial_support,epipolar_error
from marjum_transmitter_joint_polish import Polish,FREE


def run(output='cv_transmitter_joint_v4',max_nfev=200,fixed_sigma=15.):
    fit=Polish('cv_transmitter_joint_v1/fit_transmitter.npz',output)
    # Reject the failed tentative pair from the preceding geometric audit.
    fit.pairs=[p for p in fit.pairs if {p['a'],p['b']}!={'2159','2199'}]
    ranges=np.linalg.norm(fit.base[:,:3]-fit.tx0,axis=1)
    observations=np.array([fit.meta[k]['transmitter_px'] for k in FREE])
    original_unpack=fit.unpack
    def unpack(x):
        q=x[:54].reshape(6,9);tx=fit.tx0+x[54:57];cams=[]
        ks=fit.k0+q[:,5:7]
        for i in range(6):
            p=np.r_[fit.base[i,:3],fit.base[i,3:6]+q[i,1:4],fit.base[i,6]*np.exp(q[i,4])]
            d=rays(p,fit.shapes[i],[observations[i]+q[i,7:9]],ks[i])[0]
            p[:3]=tx-ranges[i]*np.exp(q[i,0])*d;cams.append(p)
        return np.array(cams),ks
    fit.unpack=unpack
    terrain_report=json.loads(Path('cv_transmitter_joint_v2/report.json').read_text())['terrain_only']
    limits=np.array([terrain_report['horizons'][k]['train_rms_px']*1.15+4 for k in FREE])
    exact_rows=[]
    def residual(x,structure=False):
        cams,ks=unpack(x);q=x[:54].reshape(6,9);tx=fit.tx0+x[54:57];values=[];deps=[];linear=[]
        def add(v,cols,strict=False):
            v=np.atleast_1d(v);values.extend(v);deps.extend([cols]*len(v));linear.extend([strict]*len(v))
        for i,key in enumerate(FREE):
            p=cams[i];cols=list(range(9*i,9*i+9))+list(range(54,57))
            h=fit.horizon(i,tuple(p),tuple(ks[i]))[fit.train[i]]
            add(h/8,cols)
            add((p[:3]-fit.base[i,:3])/[8,8,5],cols,True)
            add(q[i,1:4]/.10,cols,True);add([q[i,4]/.10],cols,True);add(q[i,5:7]/.025,cols,True)
            add(q[i,7:9]/3,cols,True)
            ground=max(fit.terrain.height(*p[:2]),float(np.asarray(fit.terrain.dem.interp_alt(np.array([p[0]]),np.array([p[1]])))[0]))
            add([min(p[2]-ground-.6,0)/.1],cols,True)
            add(np.minimum(radial_support(p,fit.shapes[i],ks[i])-.3,0)*100,cols,True)
            add([max(np.sqrt(np.mean(h*h))-limits[i],0)/.5],cols,True)
        for key in fit.txkeys:
            if key in FREE:continue
            p,s,k=fit.camera(key,cams,ks);pred,depth=project(p,s,tx,k)
            add((pred[0]-fit.meta[key]['transmitter_px'])/fixed_sigma,list(range(54,57)))
            add([min(depth[0]-.5,0)/.1],list(range(54,57)),True)
        for pair in fit.pairs:
            a,b=pair['a'],pair['b'];p,s,k=fit.camera(a,cams,ks);p2,s2,k2=fit.camera(b,cams,ks)
            cols=[j for key in [a,b] if key in FREE for j in range(9*fit.active[key],9*fit.active[key]+9)]+list(range(54,57))
            e=epipolar_error(p,s,k,p2,s2,k2,pair['x'],pair['y'])
            add(np.nan_to_num(e,nan=1000.)/3,cols)
            add([min(np.linalg.norm(p[:3]-p2[:3])-.03,0)/.01],cols,True)
        if structure:
            exact_rows.extend(linear)
            mat=lil_matrix((len(values),57),dtype=int)
            for i,cols in enumerate(deps):mat[i,cols]=1
            return mat.tocsr()
        return np.array(values)
    # Source camera rays supply a consistent initial intersection, independent
    # of the stale transmitter array in the supplied NPZ.
    A=np.zeros((3,3));b=np.zeros(3)
    for i in range(6):
        d=rays(fit.base[i],fit.shapes[i],[observations[i]],fit.k0[i])[0];n=np.eye(3)-np.outer(d,d)
        A+=n;b+=n@fit.base[i,:3]
    initial_tx=np.linalg.solve(A,b);x0=np.zeros(57);x0[54:]=initial_tx-fit.tx0
    for i in range(6):x0[9*i]=np.log(np.linalg.norm(initial_tx-fit.base[i,:3])/ranges[i])
    sparsity=residual(x0,True);strict=np.array(exact_rows)
    def loss(z):
        t=1+z;rho=np.array([2*(np.sqrt(t)-1),1/np.sqrt(t),-.5*t**(-1.5)])
        rho[:,strict]=[z[strict],np.ones(strict.sum()),np.zeros(strict.sum())]
        return rho
    bound=np.r_[np.tile([1.2,.3,.3,.25,.25,.08,.08,15,15],6),[6,6,6]]
    scale=np.r_[np.tile([.08,.02,.02,.02,.03,.01,.01,1,1],6),[1,1,1]]
    result=least_squares(residual,x0,bounds=(-bound,bound),jac_sparsity=sparsity,x_scale=scale,
                         loss=loss,f_scale=2,max_nfev=max_nfev,ftol=1e-6,verbose=2)
    tx=fit.tx0+result.x[54:];fit.save('fit_transmitter.npz',result.x,tx)
    report=fit.metrics(result.x,tx)
    report.update(source=str(fit.source),input_sha256=fit.report['input_sha256'],initializer=initial_tx.tolist(),
                  optimizer=dict(success=bool(result.success),nfev=result.nfev,cost=float(result.cost)),
                  horizon_limits_train_px=dict(zip(FREE,limits.tolist())),
                  excluded_pair=['2159','2199'],note=f'All six cameras and transmitter optimized jointly. Pixel offsets have 3px priors. Fixed camera registration uses {fixed_sigma:g}px scale. No posterior precision claimed.')
    (fit.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(fit.out/'optimizer.npz',x=result.x,jac=result.jac.toarray(),residual=result.fun)
    print('transmitter',tx,flush=True)
    return report

if __name__=='__main__':run()
