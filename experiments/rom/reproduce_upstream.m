% Run unchanged upstream Figure 8(b,c), then export compact parity data.
% Launch from a private COPY of the verified Zenodo v_2 archive.
% Original MATLAB files and the distributed Newmark traces remain unchanged.
set(groot,'defaultFigureVisible','off');
maxNumCompThreads(2);
if isempty(gcp('nocreate'))
    parpool('local',4);
end
fprintf('SIAM_ROM_REPRODUCTION_BEGIN %s\n',datestr(now,30));
run('Figure_8_bc_Plot.m');
fprintf('SIAM_ROM_FIGURE8_FINISHED %s\n',datestr(now,30));
% Figure_8 clears workspace; recover all export values from its final state.
V_export=ROM_POD.V;
M_r=V_export'*(HFM.M*V_export);
K_r=V_export'*(HFM.K*V_export);
b_r=V_export'*HFM.f;
cx_r=HFM.Rec(1:2:end)'*V_export(1:2:end,:);
cy_r=HFM.Rec(2:2:end)'*V_export(2:2:end,:);
time_grid=var_para.t;
wr=var_para.s0(1);wi=var_para.s0(2);Nz=var_para.Nz;
alpha=var_para.width;amplitude=var_para.Amp;t0=var_para.t0;
full_dimension=HFM.DOF;
weighted_orthogonality=norm(V_export'*(HFM.Xh*V_export)-eye(size(V_export,2)),'fro');
s_query=0.21+1i*[0.75,1.5,3,5];
q_query=double(eval(var_para.source(amplitude,alpha,t0,s_query)));
reduced_query=zeros(size(V_export,2),numel(s_query));
full_receivers=zeros(2,numel(s_query));
for jj=1:numel(s_query)
    reduced_query(:,jj)=(s_query(jj)^2*M_r+K_r)\(q_query(jj)*b_r);
    full_query=(s_query(jj)^2*HFM.M+HFM.K)\(q_query(jj)*HFM.f);
    full_receivers(1,jj)=HFM.Rec(1:2:end)'*full_query(1:2:end);
    full_receivers(2,jj)=HFM.Rec(2:2:end)'*full_query(2:2:end);
    fprintf('SIAM_ROM_HOLDOUT %d/%d\n',jj,numel(s_query));
end
save('reproduction_result.mat','M_r','K_r','b_r','cx_r','cy_r',...
    'time_grid','wr','wi','Nz','alpha','amplitude','t0','full_dimension',...
    'weighted_orthogonality','s_query','q_query','reduced_query',...
    'full_receivers','basis_id_wid','error_seismo_POD','-v7');
fprintf('SIAM_ROM_REPRODUCTION_COMPLETE %s\n',datestr(now,30));
delete(gcp('nocreate'));
