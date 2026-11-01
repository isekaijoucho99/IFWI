% Analytic small-system fixture executed through the ORIGINAL upstream function.
% This complements, and does not replace, the distributed-mesh reproduction.
addpath('Reduced_basis_methods');maxNumCompThreads(1);
M=diag([1,2]);K=diag([4,9]);b=[1;1];Rec=[1;1];V=eye(2);
var.t=linspace(0,8,71);var.s0=[.2,3];var.Nz=256;
theta=((-var.Nz:var.Nz-1)+.5)*pi/var.Nz;
c=exp(1i*theta);nodes=var.s0(1)-var.s0(2)*(c+1)./(c-1);
var.s_LFids=1:var.Nz;s=nodes(var.Nz+1:end);q=1./(s+1.3);
Kl={K,zeros(2),zeros(2),zeros(2),zeros(2)}';
Ku={zeros(2),zeros(2),zeros(2),zeros(2),zeros(2)}';
[seismogram,frequency,coefficients]=getseismosTD_lam(M,b,s,q,var,Rec,V,[1,0,0,0,0],zeros(1,5),Kl,Ku);
time_grid=var.t;
save('upstream_core_fixture.mat','M','K','b','V','Rec','nodes','q','seismogram','frequency','coefficients','time_grid','-v7');
fprintf('UPSTREAM_CORE_FIXTURE_COMPLETE\n');
