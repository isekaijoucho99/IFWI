# Fourier reparameterization and fixed background-relative velocity scale

Audit date: 2026-11-13. These notes distinguish upstream evidence, local code facts and untested hypotheses. They do not claim that this combination has already been validated in the literature.

## Sources and the reused implementation

[Shi, Zhou and Gu, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Improved_Implicit_Neural_Representation_with_Fourier_Reparameterized_Training_CVPR_2024_paper.html) describes learning coefficients of fixed Fourier bases to compose MLP weights. The [official FR-INR repository](https://github.com/CVL-UESTC/FR-INR) identifies itself as the paper's PyTorch implementation. Its [`sin_fr_layer` in modules.py](https://github.com/CVL-UESTC/FR-INR/blob/main/modules.py) constructs cosine bases at low frequencies (i+1)/L and high frequencies i+1, phase offsets 2*pi*p/P, and inclusive samples over the longest period. Its trainable coefficient initialization uses per-basis bounds sqrt(6/M)/(norm(basis)*omega0), with zero bias; its forward pass applies sin(omega0*(xW+b)). The [image-fitting example](https://github.com/CVL-UESTC/FR-INR/blob/main/2d_image_fitting.py) uses L=128, H=128, P=32 and recommends alpha=.01 for sine networks.

The upstream files were inspected through the public browser on the audit date. These are floating `main` links: an upstream commit identifier was not obtained from the accessible responses, so this is not a claim of a pinned upstream snapshot. The exact reused local implementation is `experiments/fourier_modules.py` from base commit `2bc95d157637f035d19473be29b30922e6dfd14b`, SHA256 `709bd5593cf73eebf8c7fd6a19ed5926343c83ca6634d17a4e7d590c192a280a`; this experiment does not edit it.

[Kang et al., *Implicit full waveform inversion with adaptive Fourier frequency bases learning*, GJI 244(1), ggaf404](https://doi.org/10.1093/gji/ggaf404) discusses Fourier reparameterization for IFWI. It was published online on 14 October 2025 with a January 2026 issue date. Its data-availability section directs readers to published datasets and requests to the authors; no GitHub link was present in the inspected article text. This local experiment is not a reproduction of that paper's full author pipeline or reported numerical results.

## Local adaptation and dimensions

The existing local code replaces only three hidden-to-hidden layers. It retains the first coordinate layer and zero-initialized final layer. It stores bases as nonpersistent buffers, computes W=Lambda@B with a plain bias vector, and leaves the original IRN sine activation in place. Bases are reconstructed from configuration on restore. Its vectorized coefficient sampling uses the author's bound, but differs from the author's per-column sampling order: identical seeds do not imply bit-identical author coefficients. Existing verification tests compare basis values and initialization bounds.

For [2,128,128,128,128,1], L=128/H=128/P=32 gives 8192 basis rows per hidden layer. Each Lambda has shape [128,8192]. There are **3,146,625 trainable entries**, compared with **50,049** for the plain network; three fixed bases contain another **3,145,728 entries**. This is not a parameter-count-matched ablation. The Fourier bases tie the first and last input columns because of inclusive periodic endpoints; the local float64 rank check is 127 at width128. More coefficients therefore do not imply an unrestricted or strictly larger effective weight space than the plain 128-column matrices.

Float32 trainable tensors occupy 12,586,500 bytes; fixed Fourier bases occupy 12,582,912 bytes. Ordinary Adam's two moment tensors occupy 25,173,000 bytes, plus ten float32 scalar steps (40 bytes) once all parameters have state. These figures exclude gradients, other buffers, temporary matrix products, velocity/wavefield graphs and allocator overhead; GPU peak memory must be measured.

## Parameterization implemented here

`RelativeVelocityParameterization` subclasses the existing velocity model and evaluates

    v_theta = v0 + relative_scale * (1000*std_kmps) * f_theta
    relative_scale = v0 / sqrt(mean(v0**2))

The one-time RMS reduction uses float64, then the scalar and persistent nontrainable scale use the background float32 dtype. The scale uses only the frozen initial background. With std_kmps=1, sv=1000m/s. The zero final layer gives v_theta=v0 exactly at update0 for both plain and Fourier networks. Existing Adam and checkpoint APIs are used; no channel gates or curriculum are present.

By the chain rule the parameter VJP uses (1000*std_kmps)*relative_scale times the physical velocity gradient before applying the network Jacobian transpose. This is an algebraic property, not a measured convergence gain. The parameterization is a fixed background-relative additive perturbation. It is not an exact exponential/log-velocity model and does not compensate every illumination, acquisition or wave-propagation effect. The composite changes the network representation and this scale together; neither individual contribution is isolated by the twelve-run comparison.

The new CPU tests check exact initial velocity, persistent fixed scale, analytic and finite-difference parameter derivatives, all-buffer immutability under Adam, the actual full-width coefficient count, and exact model/Adam/Python/NumPy/Torch RNG recovery plus the next stochastic optimizer update. Physical convergence, stability, runtime and memory remain GPU experiment outcomes, not conclusions of these tests.
