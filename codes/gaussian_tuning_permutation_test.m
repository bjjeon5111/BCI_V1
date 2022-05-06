% Computes p-value on the Rsquared of the fit using permutation of individual trials.
function [rsquare_actual,rsquare_pval] = gaussian_tuning_permutation_test(DataMatrix)
%% this function will run a permutation test on 2-D gaussian tuning to see if the generated R^2 is greater than chance R^2
%  chance R^2 is generated from
% input:
%    DataMatrix - [Nx1] cell array containing neural responses to stimuli, where each cell is the response data for one neuron.
%       Within each cell of this array, is a nested cell array of dimensions [axb]. a is the number of orientations, and b is the number of spatial frequencies(in our case this is the number of
%       orientations. Each stimulus of the nested array contains [dx1] vector, where each entry in the vector is 
%       a single trial response value.

% output:
%   rsquare_actual - [Nx1] double array containing the R^2 value of the 2-D gaussian fit.
%   rsquare_pval - [Nx1] double array of the p-value on the R^2 value.

maxperm = 1000; % number of permutations for goodness of fit testing
rsq_pval_threshold = 0.05; % p-val threshold is p<0.05

SFspacing = 0.02;
SFLow = 0.02;
SFHigh = 0.3;
numSF = 15;
numOri = 12;

responseMat = DataMatrix;
%%
numreps = length(DataMatrix{1,1}{1,1});% how many repetitions per grating
numcells = size(responseMat,1);
avgresponse = cell(size(responseMat,1),1);
orientations = zeros(numOri,numSF);
sp_freqs = zeros(numOri,numSF);
kx = numSF;
ky = numOri;
%responseMat3 = zeros(624*numreps_nophase,numcells);
%ori_rMat3 = zeros(624*numreps_nophase,1);
%spfreq_rMat3 = zeros(624*numreps_nophase,1);
for cellind = 1:size(responseMat,1)
    avgresponse{cellind,1} = zeros(ky,kx);
    stimCounter = 0;
    for i=1:kx
    for j = 1:ky
        stimCounter = stimCounter+1;       
        % this is the trial matched version, so cut off any trials after the trial limit per stim        
        avgresponse{cellind,1}(j,i) = nanmean(responseMat{cellind,1}{j,i});
        if cellind == 1
            orientations(j,i) = (j-1)*15;
            sp_freqs(j,i) = 0.02*i;
        end
    end
    end
end
 %%
% figure;
avgresponse = avgresponse_Filtered;
[pref_spfreq_estimate,pref_ori_cp,pref_ori_2dgaussfit,pref_spfreq_2dgaussfit,fit_corr_pval,fit_corr,moddepth] = deal(zeros(numcells,1));
[gauss2dfitresults,gauss2dgof] = deal(cell(numcells,1));
orientations_vec = orientations(:);
spfreqs_vec = sp_freqs(:);
[rsquare_pval,rsquare_pval_emp,rsquare_actual] = deal(zeros(numcells,1));
for i = 1:numcells
    % compute half of complex phase
    tic;
    disp(['Processing Cell ',num2str(i),'...']);
    % calculate the complex phase to see which orientation should be
    % centered in the fit
    S = sum(avgresponse{i}(:).*exp(2*1i*(orientations_vec*2*pi/360)))/sum(avgresponse{i}(:));
    complex_phase_angle = angle(S)/(2*pi)*360;
    pref_ori_cp(i) = complex_phase_angle/2;
    % shift orientation based on the calculated complex phase
    orientations_shifted = orientations_vec;
    if pref_ori_cp(i) < 90
        orientations_shifted(orientations_vec>pref_ori_cp(i)+90) = orientations_shifted(orientations_vec>pref_ori_cp(i)+90)-180;
    elseif pref_ori_cp(i) > 90
        orientations_shifted(orientations_vec<pref_ori_cp(i)-90) = orientations_shifted(orientations_vec<pref_ori_cp(i)-90)+180;
    end
    
    %% uncomment if you want to plot the tuning----------------------------
    %imagesc(0.02:0.02:0.3,0:15:165,avgresponse{i});
    %cla;
    %scatter(spfreqs_vec,orientations_shifted,72,avgresponse{i}(:),'filled','MarkerEdgeColor','k')
    %xlim([0,0.32])
    %ylim([min(orientations_shifted)-15,max(orientations_shifted)+15])
    %xlabel('Sp Freq (cpd)')
    %ylabel('Orientation (degrees)')
    %colormap('hot')
    %colorbar
    %title(['Spatial Frequency and Orientation Tuning Cell ',num2str(i)])
    %----------------------------------------------------------------------
    %%
 
    %fit a 2-D gaussian tuning
    [~,max_resp_ind] = max(avgresponse{i}(:));
    pref_spfreq_estimate(i) = spfreqs_vec(max_resp_ind);
    
    % vectorize the responses and the stimulus parameters
    response_vec = DataMatrix{i}(:);
    response_vec = cell2mat(response_vec);
    spfreqs_vec = imresize(spfreqs_vec,[numreps*numSF*numOri,1],'nearest');
    orientations_shifted = imresize(orientations_shifted,[numreps*numSF*numOri,1],'nearest');
    
    %removed the filtered trials, indicated by NaN values
    removeInd = isnan(response_vec);
    response_vec(removeInd) = [];
    orientations_shifted(removeInd) = [];
    spfreqs_vec(removeInd) = [];
    
    
    [xData,yData,zData] = prepareSurfaceData(spfreqs_vec(:),orientations_shifted,response_vec);
    ft = fittype('A*exp(-1/(2*(1-rho^2))*((x-mu1)^2/sig11^2+(y-mu2)^2/sig22^2-2*rho*(x-mu1)*(y-mu2)/(sig11*sig22)))/(2*pi*sig11*sig22*sqrt(1-rho^2))+C',...
    'dependent',{'z'},'independent',{'x','y'},...
    'coefficients',{'mu1','mu2','sig11','sig22','rho','A','C'});
    % mu1 = preferred spatial frequency
    % mu2 = preferred orientation
    % sig11 = spatial frequency width
    % sig22 = orientation width
    % rho = correlation between spfreq and orientatio
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Lower = [SFLow,pref_ori_cp(i)-90,0.001,1,-1,0,0];
    opts.Upper = [SFHigh,pref_ori_cp(i)+90,+inf,+inf,1,+inf,+inf];
    opts.StartPoint = [pref_spfreq_estimate(i),pref_ori_cp(i),0.1,45,0,max(zData),0];
    opts.MaxIter = 400;
    [gauss2dfitresults{i}, gauss2dgof{i}] = fit([xData,yData],zData, ft, opts );
    
    %% uncomment if you want to plot the fit ------------------------------
    %xrange = xlim;
    %yrange = ylim;
    %[Xgrid,Ygrid]=meshgrid([xrange(1):0.001:xrange(2)],[yrange(1):0.1:yrange(2)]);
    %fitted_surf = gauss2dfitresults{i}(Xgrid,Ygrid);
    %peak_resp_fit = max(fitted_surf(:));
    %moddepth(i) = peak_resp_fit-gauss2dfitresults{i}.C;
    %hold on
    %contour_lvls = [peak_resp_fit-0.75*moddepth(i);peak_resp_fit-0.5*moddepth(i);peak_resp_fit-0.25*moddepth(i)];
    %contour(Xgrid,Ygrid,fitted_surf,contour_lvls,'linewidth',1.5)
    %----------------------------------------------------------------------

    
        %% permutation test for goodness of fit
    % Is the rsquare value greater than expected by random chance?
    rsquare_permuted = zeros(maxperm,1);
    rng('shuffle'); % shuffle the random number gernerator seed
    parfor permind = 1:maxperm
        randomind = randperm(length(zData));
        zData_randomized = zData(randomind);
        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
        opts.Display = 'Off';
        opts.Lower = [SFLow,pref_ori_cp(i)-90,0.001,1,-1,0,0];
        opts.Upper = [SFHigh,pref_ori_cp(i)+90,+inf,+inf,1,+inf,+inf];
        opts.StartPoint = [pref_spfreq_estimate(i),pref_ori_cp(i),0.1,45,0,max(zData),0];
        opts.MaxIter = 400;
        [~,perm_test_gof] = fit([xData,yData],zData_randomized, ft, opts );
        rsquare_permuted(permind) = perm_test_gof.rsquare;
    end
    
    rsquare_actual(i) = gauss2dgof{i}.rsquare;
    rsquare_pval(i) = 1-normcdf(gauss2dgof{i}.rsquare,mean(rsquare_permuted),std(rsquare_permuted));
    % alternatively, could use the empirical distribution and compute the
    % r-sqaure, but standard is to take the cumulative distribution value
    % of the approximated normal distribution
    rsquare_pval_emp(i) = sum(rsquare_permuted>gauss2dgof{i}.rsquare)/length(rsquare_permuted);
    
    toc
end
disp('All Done!');

