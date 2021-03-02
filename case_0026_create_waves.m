load cluster_run1_1_95

% Initial parameters
magn_idx = 3:3:306;
channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);

Fs = 1/(Data.Time(2)-Data.Time(1));

PARAMS = struct();
PARAMS.max_distance = 0.04; 
PARAMS.wave.half_width = 0.005; % (m) half width of one wave
PARAMS.wave.duration = 0.02; % (s) time of wave duration 
PARAMS.wave.speeds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; % (mm/ms = m/s)
PARAMS.sampling_rate = Fs;
faces = cortex.Faces;
vertices = cortex.Vertices;

% remove bad segments
for i = 1:length(Data.Events)
    events_name{i} = Data.Events(i).label;
end

% % Forward model matrix
G = bst_gain_orient(G3.Gain, G3.GridOrient); 

ind_cl = 5;
ind_n = cluster{1,ind_cl}(1,:);
ind_spike = cluster{1,ind_cl}(2,:);

clear Ndir
PARAMS.VertConn = tess_vertconn(cortex.Vertices, cortex.Faces);
wavelin_cluster = [];
for npoint = 1:length(ind_n)
    PARAMS.seed_vertices = ind_n(npoint);
    waves = wave_on_sensors_simple_3(cortex, PARAMS, G(channel_idx,:));
    wavelin = [];
    Ndir(npoint) = size(waves,1);
    for s = 1:length(PARAMS.wave.speeds)
        wavesspeed = waves(:,s);
        clear wavesspeedlin
        for d = 1:Ndir(npoint) % for each direction
            tmp = wavesspeed{d}; % matrix (num sensors) by (num time points)
            wavesspeedlin(d,:) = tmp(:); % (num dir) by (num sensors)x(num time points)
        end
        wavelin = [wavelin, wavesspeedlin];
    end
    wavelin_cluster = [wavelin_cluster; wavelin];
    npoint
end

csvwrite('waves_cluster_new5.csv', wavelin_cluster)
csvwrite('ndir_cluster_new5.csv', Ndir)
clear wavelin_cluster


f_low = 10;
f_high = 100;
[b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
Ff = filtfilt(b, a, Data.F(channel_idx,:)')';

clear DataLin
wavelength = size(waves{1, 1}, 2);
k = 1;
R = 61 - size(waves{1, 1}, 2);
for npoint = 1:length(ind_spike)
    spike_ts = Ff(:,(ind_spike(npoint)-20):(ind_spike(npoint)+40));
    range = 1:wavelength;
    for r = 1:R % sliding time window
        tmp = spike_ts(:, range); % time interval with wave
        DataLin(:, k) = tmp(:);
        range = range+1;
        k = k+1;
    end
end

csvwrite('spike_cluster_new5.csv', DataLin)

