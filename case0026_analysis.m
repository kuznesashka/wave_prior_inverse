
% Initial parameters

MEG_channels = 1:306;
magn_idx = 3:3:306;
grad_idx = setdiff(MEG_channels, magn_idx);
channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);

Fs = 1/(Data.Time(2)-Data.Time(1));

PARAMS = struct();
PARAMS.max_distance = 0.04; 
PARAMS.wave.half_width = 0.005; % (m) half width of one wave
PARAMS.wave.duration = 0.02; % (s) time of wave duration 
PARAMS.wave.speeds = [0.001, 0.005, 0.01, 0.05, 0.1, ...
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.5]; % (mm/ms = m/s)1
PARAMS.sampling_rate = Fs;
faces = cortex.Faces;
vertices = cortex.Vertices;

% remove bad segments
for i = 1:length(Data.Events)
    events_name{i} = Data.Events(i).label;
end
% 
% if ismember('BAD', events_name)
% addit = Data.Time(1);
% num_event = find(strcmp(events_name, 'BAD'));
% bad_idx = [];
% for i = 1:size(Data.Events(num_event).times, 2)
%     bad_idx = [bad_idx, int32((Data.Events(num_event).times(1,i)-addit+0.001)*Fs):...
%         int32((Data.Events(num_event).times(2,i)-addit+0.001)*Fs)];
% end
% end

num_event_spike = find(strcmp(events_name, 'spikes'));
spike_sample = int32((Data.Events(1,num_event_spike).times-Data.Time(1))*Fs);

% Forward model matrix
G = bst_gain_orient(G3.Gain, G3.GridOrient);

Nsrc = size(G3.Gain,2)/3; % number of sources
Ns = length(channel_idx); % number of sources
range3 = 1:3;
range2 = 1:2;
G2 = zeros(Ns,2*Nsrc);
for i = 1:Nsrc
    g = G3.Gain(channel_idx,range3);
    [u s v] = svd(g, 'econ');
    G2(:,range2) = u(:,1:2);
    range3 = range3+3;
    range2 = range2+2;
end
clear G3

% Dipole fitting with the MUSIC scan, region of interest
% the whole interval with a spike
f_low = 10;
f_high = 400;
[b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
Ff = filtfilt(b, a, Data.F(channel_idx,:)')';

ind = 1;
spike_ts = Ff(:,(spike_sample(ind)-20):(spike_sample(ind)+20));

figure
plot(spike_ts')

[U,S,V] = svd(spike_ts);
num = find(cumsum(diag(S).^2)/sum(diag(S).^2)>=0.95, 1);
corr = MUSIC_scan(G2, U(:,1:num));

[ValMax, IndMax] = max(corr);

% illustration
figure
h = trimesh(faces,vertices(:,1),vertices(:,2),vertices(:,3), ones(1,size(vertices(:,1),1)));
set(h,'FaceAlpha',0.5);
axis equal
grid off
axis off
view(360, 360)
hold on
scatter3(vertices(IndMax,1),vertices(IndMax,2),vertices(IndMax,3), 100,'r', 'filled')

% Neighbourhood
% ind_n = IndMax;
% 
% DIST = zeros(1, size(vertices,1));
% for d = 1:size(vertices,1)
%     DIST(d) = norm(vertices(d,:)-vertices(IndMax,:));
% end
% 
% ind_n = find(DIST<0.01);
% 
% surf_hr = cortex;
% surf_lr = cortex;
% vert_inds_lr = [];
% vertices = cortex.Vertices;
% faces = cortex.Faces;
% 
% minmsecort = repmat(0,1,size(vertices,1));
% minmsecort(ind_n) = 1;
% data_lr = minmsecort';
% 
% h = plot_brain_cmap2(surf_hr, surf_lr, vert_inds_lr, data_lr)
% hold on
% sphere_marker(vertices(IndMax, 1),vertices(IndMax, 2),vertices(IndMax, 3), 0.002, [0,0,0])

% sparse adjacency matrix
ind_n = IndMax;
PARAMS.VertConn = tess_vertconn(cortex.Vertices, cortex.Faces);
for npoint = 1:length(ind_n)
    PARAMS.seed_vertices = ind_n(npoint);
    sensor_waves{npoint} = wave_on_sensors_simple(cortex, PARAMS, G(channel_idx,:));
    npoint
end

%
f_low = 10;
f_high = 100;
[b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
Ff = filtfilt(b, a, Data.F(channel_idx,:)')';

ind = 1;
spike_ts = Ff(:,(spike_sample(ind)-20):(spike_sample(ind)+20));

c = lines(8);
figure
plot(spike_ts', 'Color', c(1,:))
xlabel('Время, мс')
ylabel('Амплитуда сигнала, Т')

Nsens = size(Ff,1);
wavelength = PARAMS.wave.duration/(1/PARAMS.sampling_rate) + 1;
bestRsq = zeros(length(PARAMS.wave.speeds), length(ind_n));
bestshift = zeros(length(PARAMS.wave.speeds), length(ind_n));

% for one starting point
% and different lambdas

clear bestRsq bestshift B STATS
for s = 1:length(PARAMS.wave.speeds)
        npoint = 1;
        rs = 1;
        clear DataHat RSS Rsq wavesspeedlin waves wavesspeed
        waves = sensor_waves{npoint}; 
        Ndir = size(waves,1);
        wavesspeed = waves(:,s); 
        for d = 1:Ndir % for each direction
            tmp = wavesspeed{d}; % matrix (num sensors) by (num time points)
            wavesspeedlin(d,:) = tmp(:); % (num dir) by (num sensors)x(num time points)
        end

        R = size(spike_ts,2)-wavelength; % time points minus the wave length
        range = 1:wavelength;
        for r = 1:R % sliding time window
            tmp = spike_ts(:,range); % time interval with wave
            DataLin = tmp(:);
            TSS = sum((DataLin - mean(DataLin)).^2);
            [B{s,rs},STATS{s,rs}] = lasso(wavesspeedlin', DataLin,'CV', 10, 'Alpha', 0.5);
            DataHat{rs} = B{s,rs}'*wavesspeedlin +repmat(STATS{s,rs}.Intercept', 1, size(DataLin,1));
            SHIFT(rs,1) = r;
            diff = (DataHat{rs}'-repmat(DataLin,1,size(DataHat{rs},1)));
            RSS(rs,:) = sum((diff-repmat(mean(diff,1), size(diff,1), 1)).^2);
            Rsq(rs,:) = ones(1,size(DataHat{rs},1)) - RSS(rs,:)./TSS;
            range = range+1;
            rs = rs+1;
        end
      
        [bestRsq(s,:), bestshift(s,:)] = max(Rsq);
    s
end

wavelin = [];
waves = sensor_waves{npoint};
Ndir = size(waves,1);
for s = 1:length(PARAMS.wave.speeds)
    wavesspeed = waves(:,s); 
    for d = 1:Ndir % for each direction
        tmp = wavesspeed{d}; % matrix (num sensors) by (num time points)
        wavesspeedlin(d,:) = tmp(:); % (num dir) by (num sensors)x(num time points)
    end
    wavelin = [wavelin, wavesspeedlin];
end
csvwrite('waves.csv', wavelin)
% R = size(spike_ts,2)-wavelength; % time points minus the wave length
% range = 1:wavelength;
% for r = 1:R % sliding time window
%     tmp = spike_ts(:,range); % time interval with wave
%     DataLin(:,r) = tmp(:);
%     range = range+1;
% end
% csvwrite('spike.csv', DataLin)



for j = 1:100
    for i = 1:length(PARAMS.wave.speeds)
        bestdf(i, j) = sum(B{i, bestshift(i,j)}(:,j)~=0);
    end
end


figure
subplot(2,1,1)
imagesc(bestRsq)
colorbar
ylabel('Speeds of wave propagation')
xlabel('Regularization parameter lambda')
title('R-squared values')
ax = gca;
ax.YTick = 1:length(PARAMS.wave.speeds);
ax.YTickLabel = (PARAMS.wave.speeds);
subplot(2,1,2)
imagesc(bestdf)
colorbar
ylabel('Speeds of wave propagation')
xlabel('Regularization parameter lambda')
title('Number of nonzero coefficients')
ax = gca;
ax.YTick = 1:length(PARAMS.wave.speeds);
ax.YTickLabel = (PARAMS.wave.speeds);


lambda = 46;
figure
subplot(2,1,1)
stem(PARAMS.wave.speeds, bestRsq(:,lambda), 'LineWidth', 2)
subplot(2,1,2)
stem(PARAMS.wave.speeds, bestdf(:,lambda), 'LineWidth', 2)

% diagram of directions used

[valsort indsort] = sort(bestRsq(:,lambda), 'descend');
[val ind] = sort(bestdf(indsort(1:3),lambda));
bestind = indsort(ind(1));

bestspeed = PARAMS.wave.speeds(bestind);
bestcoef = B{bestind,bestshift(bestind,lambda)}(:,lambda);

% picture
theta = 0:0.01:pi/4;
[val, ind] = sort(abs(bestcoef), 'descend');
bestcoef2 = bestcoef(ind);

figure
for i = 1:5
    if ind(i) ~=5
        rho = sin(2*theta).*cos(2*theta)*2*abs(bestcoef2(i));
        theta2 = theta+pi/8+pi/2*(ind(i)-3);
        if bestcoef2(i)>=0
            col = 'r';
        else col = 'b';
        end
        polar(theta2, rho, col)
        hold on
    else th = linspace(0,2*pi,50);
        r = abs(bestcoef2(i));
        if bestcoef2(i)>=0
            col = 'r';
        else col = 'b';
        end
  
    polar(th,r+zeros(size(th)), col)
    hold on
    end
end

csvwrite('waves.csv', wavesspeedlin)









% for different starting cortical sources
% for one lambda
for s = 1:length(PARAMS.wave.speeds)
    for npoint = 1:length(ind_n)
        rs = 1;
        clear DF SHIFT B STATS DataHat RSS Rsq wavesspeedlin waves wavesspeed
        waves = sensor_waves{npoint}; 
        Ndir = size(waves,1);
        wavesspeed = waves(:,s); 
        for d = 1:Ndir % for each direction
            tmp = wavesspeed{d}; % matrix (num sensors) by (num time points)
            wavesspeedlin(d,:) = tmp(:); % (num dir) by (num sensors)x(num time points)
        end

        R = size(spike_ts,2)-wavelength; % time points minus the wave length
        range = 1:wavelength;
        for r = 1:R % sliding time window
            tmp = spike_ts(:,range); % time interval with wave
            DataLin = tmp(:);
            TSS = sum((DataLin - mean(DataLin)).^2);
            %lbd = lambda(wavesspeedlin', DataLin, 'CV', 10, 'Alpha', 0.5);
            %[B{rs},STATS{rs}] = lasso(wavesspeedlin',DataLin,'CV', 10, 'Alpha', 0.5, 'Lambda', lbd(40));
            [B{rs},STATS{rs}] = lasso(wavesspeedlin', DataLin,'CV', 10, 'Alpha', 0.5);
            DataHat{rs} = B{rs}'*wavesspeedlin +repmat(STATS{rs}.Intercept', 1, size(DataLin,1));
            SHIFT(rs,1) = r;
            diff = (DataHat{rs}'-repmat(DataLin,1,size(DataHat{rs},1)));
            RSS(rs,:) = sum((diff-repmat(mean(diff,1), size(diff,1), 1)).^2);
            Rsq(rs,:) = ones(1,size(DataHat{rs},1)) - RSS(rs,:)./TSS;
            range = range+1;
            rs = rs+1;
        end
      
        [bestRsq(s, npoint), bestshift(s, npoint)] = max(Rsq);
    end

    s
end

figure
stem(PARAMS.wave.speeds, bestRsq)
xlabel('Speed of propagation, m/s')
ylabel('R-squared')
title('Magnetometers, first spike, 10-100 bandpass')

% load bestRsq_tot
% load ind_n_tot
% load cortex_infl
% 
ind_n = ind_n_tot;

s = 10;
%load cortex_infl
surf_hr_inf = cortex_infl2;
surf_hr = cortex;
surf_lr = cortex;
vert_inds_lr = [];
vertices = cortex.Vertices;
minmsecort = repmat(min(bestRsq(s,:)),1,size(vertices,1));
minmsecort(ind_n) = bestRsq(s,:);
data_lr = minmsecort';

% 208336 (747) s = 0.2
[val, idx] = max(bestRsq(s,:));
h = plot_brain_cmap2(surf_hr, surf_lr, vert_inds_lr, data_lr)
colorbar
hold on
scatter3(vertices(ind_n(idx),1),vertices(ind_n(idx),2),vertices(ind_n(idx),3), 50,'r', 'filled')

h = plot_brain_cmap2(surf_hr_inf, surf_lr, vert_inds_lr, data_lr)
colorbar
hold on
scatter3(cortex_infl.Vertices(ind_n(idx),1),cortex_infl.Vertices(ind_n(idx),2),cortex_infl.Vertices(ind_n(idx),3), 50,'r', 'filled')



%
Nsens = size(Ff,1);
wavelength = PARAMS.wave.duration/(1/PARAMS.sampling_rate) + 1;
npoint = 1;
rs = 1;
clear DF MSE SPEED SHIFT B STATS DataHat RSS Rsq
for s = 1:length(PARAMS.wave.speeds);
    waves = sensor_waves{npoint}; 
    Ndir = size(waves,1);
    wavesspeed = waves(:,s); 
    for d = 1:Ndir % for each direction
        tmp = wavesspeed{d}; % matrix (num sensors) by (num time points)
        wavesspeedlin(d,:) = tmp(:); % (num dir) by (num sensors)x(num time points)
    end

    R = size(Data,2)-wavelength; % time points minus the wave length
    range = 1:wavelength;
    for r = 1:R % sliding time window
        tmp = Data(:,range); % time interval with wave
        DataLin = tmp(:);
        TSS = sum((DataLin - mean(DataLin)).^2);
        [B{rs},STATS{rs}] = lasso(wavesspeedlin(1:Ndir,:)',DataLin,'CV', 10, 'Alpha', 0.5);
        DataHat{rs} = B{rs}'*wavesspeedlin+repmat(STATS{rs}.Intercept', 1, size(DataLin,1));
        MSE(rs,:) = STATS{rs}.MSE;
        DF(rs,:) = STATS{rs}.DF;
        SPEED(rs) = PARAMS.wave.speeds(s);
        SHIFT(rs,1) = r;
        diff = (DataHat{rs}'-repmat(DataLin,1,size(DataHat{rs},1)));
        RSS(rs,:) = sum((diff-repmat(mean(diff,1), size(diff,1), 1)).^2);
        Rsq(rs,:) = ones(1,size(DataHat{rs},1)) - RSS(rs,:)./TSS;

        range = range+1;
        rs = rs+1;
    end
    
end
    

clear MaxRsq MaxRsqShift MinDF
range  = 1:R;
lambda = 1;
for s = 1:length(PARAMS.wave.speeds)
    [MaxRsq(s) MaxRsqShift(s)] = max(Rsq(range,lambda));
     MaxDF(s) = DF(range(MaxRsqShift(s)),lambda);
    range= range+R;
end;

figure
subplot(2,1,1)
plot(PARAMS.wave.speeds, MaxRsq, 'LineWidth', 1.5)
xlabel('Speed of propagation')
ylabel('Ratio of variance explained')
subplot(2,1,2)
plot(PARAMS.wave.speeds, MaxDF, 'LineWidth', 1.5)
xlabel('Speed of propagation')
ylabel('Degrees of freedom')

figure
imagesc(Rsq)

[valsort indsort] = sort(MaxRsq, 'descend');
bestspeed = PARAMS.wave.speeds(indsort(1));
bestshift = MaxRsqShift(indsort(1));
bestcoef = B{(indsort(1)-1)*R+bestshift}(:,30);
bestRsq = MaxRsq(indsort(1));

% picture
theta = 0:0.01:pi/4;
[val, ind] = sort(abs(bestcoef), 'descend');
bestcoef2 = bestcoef(ind);

figure
for i = 1:5
    if ind(i) ~=5
        rho = sin(2*theta).*cos(2*theta)*2*abs(bestcoef2(i));
        theta2 = theta+pi/8+pi/2*(ind(i)-3);
        if bestcoef2(i)>=0
            col = 'r';
        else col = 'b';
        end
        polar(theta2, rho, col)
        hold on
    else th = linspace(0,2*pi,50);
        r = abs(bestcoef2(i));
        if bestcoef2(i)>=0
            col = 'r';
        else col = 'b';
        end
  
    polar(th,r+zeros(size(th)), col)
    hold on
    end
end





