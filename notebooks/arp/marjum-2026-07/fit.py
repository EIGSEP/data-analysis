import numpy as np

def svd_fatmat(A):
    """
    Computes the economy SVD of a fat matrix A (r x c, with c >> r)
    using the eigen-decomposition of A A^T.
    
    Returns:
      U : (r x r) matrix of left singular vectors.
      S : (r,) vector of singular values.
      Vt: (r x c) matrix of right singular vectors (transposed).
    """
    AAT = A @ A.T
    eigvals, U = np.linalg.eigh(AAT)
    sign = np.sign(eigvals)
    eigvals *= sign
    U *= sign[None, :]
    idx = np.argsort(eigvals)[::-1] # Sort the eigenvalues and eigenvectors in descending order
    S = np.sqrt(eigvals[idx])
    U = U[:, idx]
    SV = A.T @ U
    return U, S, SV.T


class FatFitter:
    def __init__(self, neig_gsm=7, neig_21cm=4, neig_beam=8,
                 nside_gsm=16, nside_beam=16, real_dtype=DEFAULT_REAL_DTYPE):
        self.neig_gsm = neig_gsm
        self.neig_21cm = neig_21cm
        self.neig_beam = neig_beam
        self.nside_gsm = nside_gsm
        self.nside_beam = nside_beam
        self.real_dtype = real_dtype

    def set_consts(self, eq2top_m, crd_eq, beam_data, Isky, Tgnd, rot_ms, U_beam, U_sky, basis_out, bm_T21cm_out, hist_bins, flat_eig, terrain_mask):
        self.consts = dict(
            eq2top_m=jnp.asarray(eq2top_m[T0::DT], dtype=real_dtype),
            crd_eq=jnp.asarray(crd_eq, dtype=real_dtype),
            beam_data=prms0['beam_data'],
            Isky=prms0['Isky'],
            #Isky=jnp.asarray(sky_emdl, dtype=real_dtype),  # add monopole 21cm in gsm basis
            Tgnd=prms0['Tgnd'],
            rot_ms=jnp.asarray(rot_ms[R0::DR], dtype=real_dtype),
            #meas_data=jnp.asarray(data[T0::DT,R0::DR,F0::DF], dtype=real_dtype),
            U_beam=jnp.asarray(U_beam[F0::DF], dtype=real_dtype),
            U_sky=jnp.asarray(U_gsm[F0::DF], dtype=real_dtype),
            basis_out=jnp.asarray(U_beam[F0::DF][:,:,None] * U_gsm[F0::DF][:,None,:], dtype=real_dtype),  # precompute
            bm_T21cm_out=jnp.asarray(U_beam[F0::DF][:,:,None] * U_21cm[F0::DF][:,None,:], dtype=real_dtype),  # precompute
            hist_bins=jnp.arange(npix+1)-0.5, # precompute
            flat_eig=jnp.asarray(np.ones_like(freqs).dot(U_gsm), dtype=real_dtype),
            terrain_mask=jnp.asarray(terrain_mask, dtype=real_dtype),
        )

    def gen_crd_eq(self):
        ex, ey, ez = healpy.pix2vec(self.nside_gsm, np.arange(healpy.nside2npix(self.nside_gsm)))
        crd_eq = np.array([ex, ey, ez], dtype=self.real_dtype)
        return crd_eq

    def pack_prms(self, pdict):
        #p = np.concatenate([pdict['Isky'].ravel(), pdict['beam_data'].ravel(), pdict['T21cm_res'].ravel(), np.array([pdict['Tgnd']])])
        p = np.concatenate([pdict['Isky'].ravel(), pdict['T21cm_res'].ravel(), np.array([pdict['Tgnd']])])
        return p

    def unpack_prms(self, parr):
        Isky_fit = parr[:nsky].reshape(sky_emdl.shape)
        #beam_fit = parr[nsky:nsky+nbeam].reshape(beam_emdl.shape)
        T21cm_fit = parr[nsky+nbeam:nsky+nbeam+n21cm]
        Tgnd_fit = parr[-1]
        #return dict(Isky=Isky_fit, beam_data=beam_fit, T21cm_res=T21cm_fit, Tgnd=Tgnd_fit)
        return dict(Isky=Isky_fit, T21cm_res=T21cm_fit, Tgnd=Tgnd_fit)

    def update_constants(consts, prms):
        #consts['beam_data'] = prms['beam_data'].copy()
        consts['Isky'] = prms['Isky'].copy()
        consts['Tgnd'] = prms['Tgnd'].copy()
        return consts
        
    def pack_A(A, offset, chunk, Asky, Abm, A21cm, Agnd, wgt=1):
        Asky = np.array(Asky.reshape(-1, chunk)).T
        nsky = Asky.shape[-1]
        #Abm = np.array(Abm.reshape(-1, chunk)).T
        #nbeam = Abm.shape[-1]
        A21cm = np.array(A21cm)
        n21cm = A21cm.shape[-1]
        Agnd = np.array(Agnd)
        A[offset:offset+chunk, :nsky] = wgt * Asky
        #A[offset:offset+chunk, nsky:nsky+nbeam] = wgt * Abm
        A[offset:offset+chunk, nsky+nbeam:nsky+nbeam+n21cm] = wgt * A21cm
        A[offset:offset+chunk, -1] = wgt * Agnd
        return A
    
prms_dict = {k: v.copy() for k, v in prms0.items()}

#eps = 1.0 / (2 * MEMORY)  # assumes inv variance noise weight
#wgt = 1 / noise_lev**2
wgt = 1
#eps = 1e-9 * wgt**2
#eps = 1e3 * wgt**2
eps = 1e-8 * wgt**2

plt.figure()
#for batch in tqdm.tqdm(range(NBATCH)):
print(prms[-1])
for batch in tqdm.tqdm(range(3)):
    _times = [time.time()]
    tinds = np.random.randint(ntimes, size=MEMORY)
    rinds = np.random.randint(nrots, size=MEMORY)
    for i in range(MEMORY // FQ_CHUNK):
        finds = np.random.randint(nfreqs, size=FQ_CHUNK)
        bi = MEMORY + FQ_CHUNK*i
        # used for making "true" measurement
        _A_sky0, _A_gnd0, _A_21cm0 = get_A_sky(consts0, hpm.int_dtype(NSIDE), tinds[i], rinds[i], finds)
        _A_beam0 = get_A_beam(consts0, hpm.int_dtype(NSIDE), tinds[i], rinds[i], finds)
        Atrue = pack_A(Atrue, bi-MEMORY, FQ_CHUNK, _A_sky0, _A_beam0, _A_21cm0, _A_gnd0, wgt)
        
        _A_sky, _A_gnd, _A_21cm = get_A_sky(consts, hpm.int_dtype(NSIDE), tinds[i], rinds[i], finds)
        _A_beam = get_A_beam(consts, hpm.int_dtype(NSIDE), tinds[i], rinds[i], finds)
        pack_A(_Abuf, bi, FQ_CHUNK, _A_sky0, _A_beam0, _A_21cm0, _A_gnd0, wgt) # XXX
        #_Abuf = pack_A(_Abuf, bi, FQ_CHUNK, _A_sky, _A_beam, _A_21cm, _A_gnd, wgt)
        
        #_Abuf[bi:bi+FQ_CHUNK, :nsky] = wgt * np.array(_A_sky.reshape(-1, FQ_CHUNK)).T
        #_Abuf[bi:bi+FQ_CHUNK, nsky:nsky+nbeam] = wgt * np.array(_A_beam.reshape(-1, FQ_CHUNK)).T
        #_Abuf[bi:bi+FQ_CHUNK, nsky+nbeam:nsky+nbeam+n21cm] = wgt * np.array(_A_21cm)
        #_Abuf[bi:bi+FQ_CHUNK, -1] = wgt * np.array(_A_gnd)
        #noise = noise_lev * np.random.normal(size=MEMORY) # should make sure same noise for same meas
        #_ybuf[MEMORY:] = _Abuf[MEMORY:].dot(prms_true) + wgt * noise # fake measurements
        #_ybuf[MEMORY:] = _Abuf[MEMORY:].dot(prms_true) # fake measurements
        _ybuf[MEMORY:] = Atrue.dot(prms_true) # fake measurements
        #_ybuf[MEMORY:] = wgt * noise # fake measurements
    _times.append(time.time())
    U, S, SVt = svd_fatmat(_Abuf)
    _times.append(time.time())
    plt.semilogy(S**2)
    yr = U.T.dot(_ybuf.ravel())
    dy = yr - SVt.dot(prms)
    prms += np.dot(dy / (S**2 + eps), SVt)
    print(prms[-1], np.linalg.norm(dy) / np.linalg.norm(yr), np.linalg.norm(yr - SVt.dot(prms)) / np.linalg.norm(yr))
    consts = update_constants(consts, unpack_prms(prms))
    _Abuf[:MEMORY] = SVt[:MEMORY]
    _times.append(time.time())
    _ybuf[:MEMORY] = _Abuf[:MEMORY] @ prms
    #print([t - _times[0] for t in _times[1:]])
plt.semilogy(np.ones_like(S) * eps, 'k:')



nfreqs = consts0['U_sky'].shape[0]
ntimes = consts0['eq2top_m'].shape[0]
nrots = consts0['rot_ms'].shape[0]
prms_true = pack_prms(prms0)
nsky = prms0['Isky'].size
#nbeam = prms0['beam_data'].size
nbeam = 0
n21cm = prms0['T21cm_res'].size
nprms = prms_true.size

prms_dict['Tgnd'] = np.asarray(300.0, dtype=real_dtype)  # K
#prms_dict['beam_data'] = 1*beam_emdl #+ np.random.normal(scale=0.1*np.abs(beam_emdl), size=beam_emdl.shape)
prms_dict['Isky'] = np.asarray(0*sky_emdl + np.random.normal(scale=0.1*np.abs(sky_emdl), size=sky_emdl.shape), dtype=real_dtype)
prms_dict['T21cm_res'] *= 0
consts = {k: v.copy() for k, v in consts0.items()}
consts = update_constants(consts, prms_dict)
prms = pack_prms(prms_dict)
#noise_lev = 10.0  # K
noise_lev = 1.0  # K

#MEMORY, FQ_CHUNK, NBATCH = 512, 4, 10  # same as below
MEMORY, FQ_CHUNK, NBATCH = 384, 4, 20  # best balance?
#MEMORY, FQ_CHUNK, NBATCH = 256, 4, 30
#MEMORY, FQ_CHUNK, NBATCH = 128, 4, 60
#MEMORY, FQ_CHUNK, NBATCH = 64, 4, 120  # worse
_Abuf = np.zeros((2 * MEMORY, prms.size))
_ybuf = np.zeros((2 * MEMORY, ))
Atrue = np.zeros((MEMORY, prms.size))
print(_Abuf.size)
