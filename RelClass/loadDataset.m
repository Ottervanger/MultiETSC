function [tr_d,tr_l,ts_d,ts_l] = loadDataset(name,s)
%%
if ~exist('s','var')
    % create independent random stream
    s = RandStream.create('mrg32k3a','NumStreams',1,'Seed',0);
end

switch(lower(name))
    
%% time-series
    case 'acclima' % dataset that I created for testing
       load('data/acclima'); % 88 total data points
       % split into 11 groups (8 in each)
       ind = crossvalind('Kfold',88,11);
       labels(labels==2) = -1;
       % find a group for the test data that contains both groups
       for n=1:11,
           e(n) = min( hist( labels(ind==n), [1 2] ) ); % # of entries of least-represented class
       end
       [x,mxind] = max(e);
       ts_d = data( ind==mxind, :);
       ts_l = labels( ind==mxind );
       tr_d = data( ind~=mxind, : );
       tr_l = labels( ind~=mxind );
%        permidx = 1:size(tr_d,2);
%        permidx = fliplr(permidx);
%        ts_d = ts_d(:,permidx);
%        tr_d = tr_d(:,permidx);
       
    case 'medicalimages'
        test = load('data/MedicalImages_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/MedicalImages_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'chlorineconcentration'
        test = load('data/ChlorineConcentration_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/ChlorineConcentration_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'cinc_ecg_torso'
        test = load('data/CinC_ECG_torso_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/CinC_ECG_torso_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'cricket_x'
        test = load('data/Cricket_X_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Cricket_X_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'cricket_y'
        test = load('data/Cricket_Y_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Cricket_Y_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'cricket_z'
        test = load('data/Cricket_Z_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Cricket_Z_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'diatomsizereduction'
        test = load('data/DiatomSizeReduction_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/DiatomSizeReduction_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'facesucr'
        test = load('data/FacesUCR_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/FacesUCR_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'noninvasivefetalecg_thorax1'
        test = load('data/NonInvasiveFatalECG_Thorax1TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/NonInvasiveFatalECG_Thorax1TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'noninvasivefetalecg_thorax2'
        test = load('data/NonInvasiveFatalECG_Thorax2TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/NonInvasiveFatalECG_Thorax2TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'haptics'
        test = load('data/Haptics_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Haptics_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'symbols'
        test = load('data/Symbols_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Symbols_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'uwavegesturelibrary_x'
        test = load('data/uWaveGestureLibrary_X_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/uWaveGestureLibrary_X_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'uwavegesturelibrary_y'
        test = load('data/uWaveGestureLibrary_Y_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/uWaveGestureLibrary_Y_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'uwavegesturelibrary_z'
        test = load('data/uWaveGestureLibrary_Z_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/uWaveGestureLibrary_Z_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'wordsynonyms'
        test = load('data/WordsSynonyms_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/WordsSynonyms_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'mallat'
        test = load('data/MALLAT_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/MALLAT_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'starlightcurves'
        test = load('data/StarLightCurves_TEST');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/StarLightCurves_TRAIN');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'inlineskate'
        test = load('data/InlineSkate_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/InlineSkate_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);

    case 'control'
        test = load('data/synthetic_control_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/synthetic_control_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'motestrain'
        test = load('data/MoteStrain_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/MoteStrain_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'twoleadecg'
        test = load('data/TwoLeadECG_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/TwoLeadECG_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'sonyaiborobotsurface'
        test = load('data/SonyAIBORobotSurface_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/SonyAIBORobotSurface_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'sonyaiborobotsurfaceii'
        test = load('data/SonyAIBORobotSurfaceII_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/SonyAIBORobotSurfaceII_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'italypowerdemand'
        test = load('data/ItalyPowerDemand_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/ItalyPowerDemand_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'ecgfivedays'
        test = load('data/ECGFiveDays_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/ECGFiveDays_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'cbf'
        test = load('data/CBF_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/CBF_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'adiac'
        test = load('data/Adiac_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Adiac_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'beef'
        test = load('data/Beef_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Beef_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'faceall'
        test = load('data/FaceAll_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/FaceAll_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'coffee'
        test = load('data/Coffee_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Coffee_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'fish'
        test = load('data/FISH_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/FISH_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'trace'
        test = load('data/Trace_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Trace_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'lightning2'
        test = load('data/Lighting2_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Lighting2_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'yoga'
        test = load('data/yoga_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/yoga_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'swedishleaf'
        test = load('data/SwedishLeaf_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/SwedishLeaf_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'osuleaf'
        test = load('data/OSULeaf_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/OSULeaf_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'twop'
        test = load('data/Two_Patterns_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Two_Patterns_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'oliveoil'
        test = load('data/OliveOil_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/OliveOil_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'lightning7'
        test = load('data/Lighting7_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Lighting7_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'ecg'
        test = load('data/ECG200_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/ECG200_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'gunpoint'
        test = load('data/Gun_Point_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/Gun_Point_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'facefour'
        test = load('data/FaceFour_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/FaceFour_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'wafer'
        test = load('data/wafer_TEST.txt');
        ts_l = test(:,1);
        ts_d = test(:,2:end);
        train = load('data/wafer_TRAIN.txt');
        tr_l = train(:,1);
        tr_d = train(:,2:end);
        
    case 'profile',
        all = load('data/PROFILEDATA.mat');
        % partition
        Ntrain = 250;
        f1 = find(all.labels==1,Ntrain,'first');
        f2 = find(all.labels==-1,Ntrain,'first');
        tr_d = [all.data(f1,:); all.data(f2,:)];
        tr_l = [all.labels(f1); all.labels(f2)];
        all.data(f1,:) = []; all.data(f2,:) = [];
        all.labels(f1) = []; all.labels(f2) = [];
        ts_d = all.data;
        ts_l = all.labels;
        
    case 'natedata'
        all = csvread('data/Natedata.csv');
        labels = all(:,1);
        data = all(:,2:end);
        clear all;
        ind = crossvalind('Kfold',length(labels),11);
        labels(labels==2) = -1;
        % find a group for the test data that contains both groups
        for n=1:11,
            e(n) = min( hist( labels(ind==n), [1 2] ) ); % # of entries of least-represented class
        end
        [x,mxind] = max(e);
        ts_d = data( ind==mxind, :);
        ts_l = labels( ind==mxind );
        tr_d = data( ind~=mxind, : );
        tr_l = labels( ind~=mxind );
        
    case 'gaussdata'
        all = csvread('data/GaussData.csv');
        labels = all(:,1);
        data = all(:,2:end);
        clear all;
        ind = crossvalind('Kfold',length(labels),11);
        labels(labels==2) = -1;
        % find a group for the test data that contains both groups
        for n=1:11,
            e(n) = min( hist( labels(ind==n), [1 2] ) ); % # of entries of least-represented class
        end
        [x,mxind] = max(e);
        ts_d = data( ind==mxind, :);
        ts_l = labels( ind==mxind );
        tr_d = data( ind~=mxind, : );
        tr_l = labels( ind~=mxind );
        
        
    case 'warblers',
        alldata = importdata('data/primaryRaw.csv');
        % parse
        data = alldata.data(2:end,:); % skip 1st row
        t_labels = alldata.rowheaders(2:end,:);
        classes = unique(t_labels);
        labels = zeros(size(t_labels));
        for n=1:length(classes),
            f = find( strcmp(t_labels, classes(n)) );
            labels(f) = n;
        end
        % partition
        Ntrain = 120;
        train_ind = randsample(s,1:length(labels),Ntrain);
        tr_d = data(train_ind,:);
        tr_l = labels(train_ind);
        ts_d = data; ts_d(train_ind,:) = [];
        ts_l = labels; ts_l(train_ind,:) = [];

    case 'birds',
        alldata = importdata('data/auxiliaryDataset.csv');
        % parse
        data = alldata.data(2:end,:); % skip 1st row
        t_labels = alldata.rowheaders(2:end,:);
        classes = unique(t_labels);
        labels = zeros(size(t_labels));
        for n=1:length(classes),
            f = find( strcmp(t_labels, classes(n)) );
            labels(f) = n;
        end
        % partition
        Ntrain = 600;
        train_ind = randsample(s,1:length(labels),Ntrain);
        tr_d = data(train_ind,:);
        tr_l = labels(train_ind);
        ts_d = data; ts_d(train_ind,:) = [];
        ts_l = labels; ts_l(train_ind,:) = [];

 %% regular
    case 'heart',
        alldata = importdata('data/heart.dat');
        % parse
        data = alldata(:,1:end-1);
        labels = alldata(:,end);
        labels = sign( labels - 1.5 ); % convert 1->-1 and 2->1
        % partition
        Ntrain = 240; % 1:9 ratio
        train_ind = randsample(s,1:length(labels),Ntrain);
        tr_d = data(train_ind,:);
        tr_l = labels(train_ind);
        ts_d = data; ts_d(train_ind,:) = [];
        ts_l = labels; ts_l(train_ind,:) = [];
        
    case 'ionosphere',        
        % read
        fid = fopen('data/ionosphere.data');
        % 34 numbers, then 'g' or 'b' for class label
        fmt = '';
        for n=1:34,
            fmt = [fmt '%n '];
        end
        fmt = [fmt '%c'];
        C = textscan(fid,fmt,'Delimiter',',');
        fclose(fid);
        % parse
        data = cell2mat( C(1:end-1) );
        labels = double(C{end}=='g') - double(C{end} == 'b');
        % partition
        Ntrain = 311; % 9:1 ratio
        train_ind = randsample(s,1:length(labels),Ntrain);
        tr_d = data(train_ind,:);
        tr_l = labels(train_ind);
        ts_d = data; ts_d(train_ind,:) = [];
        ts_l = labels; ts_l(train_ind,:) = [];


    case 'sonar',
        % read
        fid = fopen('data/sonar.all-data');
        % 60 numbers, then 'R' or 'M' for class label
        % 34 numbers, then 'g' or 'b' for class label
        fmt = '';
        for n=1:60,
            fmt = [fmt '%n '];
        end
        fmt = [fmt '%c'];
        C = textscan(fid,fmt,'Delimiter',',');
        fclose(fid);
        % parse
        data = cell2mat( C(1:end-1) );
        labels = double(C{end}=='M') - double(C{end} == 'R');
        % partition
        Ntrain = 185; % 9:1 ratio
        train_ind = randsample(s,1:length(labels),Ntrain);
        tr_d = data(train_ind,:);
        tr_l = labels(train_ind);
        ts_d = data; ts_d(train_ind,:) = [];
        ts_l = labels; ts_l(train_ind,:) = [];
        
    case {'optical','optdigits','ocr'}
        % choose 3 and 8
        data = csvread('data\optdigits.tra');
        tr_d = data(:, 1:end-1 );
        tr_l = data(:,end);
        f3 = find(tr_l==3);
        f8 = find(tr_l==8);
        tr_d = tr_d([f3; f8],:);
        tr_l = [ones(numel(f3),1); -1*ones(numel(f8),1)];
        
        data = csvread('data\optdigits.tes');
        ts_d = data(:, 1:end-1 );
        ts_l = data(:,end);
        f3 = find(ts_l==3);
        f8 = find(ts_l==8);
        ts_d = ts_d([f3; f8],:);
        ts_l = [ones(numel(f3),1); -1*ones(numel(f8),1)];
        
    case {'synth'}
        Ntrain = 90; %450;
        Ntest = 10; %50;
        dim = 5;
        Total = Ntrain + Ntest;
        mag = rand(Total,1);
        class = sign( rand(Total,1)-0.5);
        mag(class==-1) = mag(class==-1) + 2;
        
        data = rand(Total,dim)-0.5;
        data = bsxfun( @times, data, mag ./ sqrt( sum(data.^2,2) ) );
        
        tr_d = data(1:Ntrain,:);
        tr_l = class(1:Ntrain);
        ts_d = data(Ntrain+1:end,:);
        ts_l = class(Ntrain+1:end);

        
       
    otherwise
        error('dataset not found');
end