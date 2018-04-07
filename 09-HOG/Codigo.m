% EXERCISE2
clear all
close all
clc
 
setup ;

% Import data
%% 

% Construct positive data
names = dir('data/myPositives/*.jpg') ;
names = fullfile('data', 'myPositives', {names.name}) ;
for i=1:numel(names)
  im = imread(names{i}) ;
  im = imresize(im, [64 64]) ;
  trainBoxes(:,i) = [0.5 ; 0.5 ; 64.5 ; 64.5] ;
  trainBoxPatches{i} = im2single(im) ;
  trainBoxImages{i} = names{i} ;
  trainBoxLabels(i) = 1 ;
end
trainBoxPatches = cat(4, trainBoxPatches{:}) ;
%%
figure(1) ; clf ;

subplot(1,2,1) ;
imagesc(vl_imarraysc(trainBoxPatches)) ;
axis off ;
title('Training images (positive samples)') ;
axis equal ;

subplot(1,2,2) ;
imagesc(mean(trainBoxPatches,4)) ;
box off ;
title('Average') ;
axis equal ;

%% 

hogCellSize = 8 ;
trainHog = {} ;
for i = 1:size(trainBoxPatches,4)
  trainHog{i} = vl_hog(trainBoxPatches(:,:,:,i), hogCellSize) ;
end
trainHog = cat(4, trainHog{:}) ;
w = mean(trainHog, 4) ;
figure(2) ; clf ;
imagesc(vl_hog('render', w)) ;

%%

pos = trainHog ;

%% 
trainDir = dir('data/TrainImages') ;
names0= fullfile('data','TrainImages',{trainDir.name},'*.jpg');
names1= fullfile('data','TrainImages',{trainDir.name});
namef={};
jusNam={};
for i=3:length(names0)
    names2=dir(names0{i});
    jusNam=cat(2,jusNam,names2.name);
    names2= fullfile(names1{i},{names2.name});
    namef=cat(2,namef,names2);
end

trainImages=namef;
%%


negDir=dir('data/myNegatives/*.JPEG') ;
negDir=fullfile('data', 'myNegatives', {negDir.name}) ;
% modelWidth = size(trainHog, 2) ;
% modelHeight = size(trainHog, 1) ;
% for i=3:numel(neg)
%   % Get the HOG features of a training image
%   t = imread(neg{i}) ;
%   t = imresize(t,[64,64]);
%   t = im2single(t) ;  
%   hog{i} = vl_hog(t, hogCellSize) ;
% end
% neg = cat(4, hog{:}) ;

% Collect negative training data
neg = {} ;
modelWidth = size(trainHog, 2) ;
modelHeight = size(trainHog, 1) ;
for t=1:numel(negDir)
  % Get the HOG features of a training image
  t = imread(negDir{t}) ;
  t = im2single(t) ;
  hog = vl_hog(t, hogCellSize) ;
  
  % Sample uniformly 5 HOG patches
  % Assume that these are negative (almost certain)
  width = size(hog,2) - modelWidth + 1 ;
  height = size(hog,1) - modelHeight + 1 ;
  index = vl_colsubset(1:width*height, 10, 'uniform') ;

  for j=1:numel(index)
    [hy, hx] = ind2sub([height width], index(j)) ;
    sx = hx + (0:modelWidth-1) ;
    sy = hy + (0:modelHeight-1) ;
    neg{end+1} = hog(sy, sx, :) ;
  end
end
neg = cat(4, neg{:}) ;


%% SVM
numPos = size(pos,4) ;
numNeg = size(neg,4) ;

x = cat(4, pos, neg) ;
x = reshape(x, [], numPos + numNeg) ;

% Create a vector of binary labels
y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;


C = 20 ;
lambda = 1 / (C * (numPos + numNeg)) ;

w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;
w = single(reshape(w, modelHeight, modelWidth, [])) ;
%%
figure(2) ; clf ;
imagesc(vl_hog('render', single(w))) ;

%%
minScale =-1;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;
% 
%  for i= 1:1
%     
im=imread(trainImages{580}) ;
im=im2single(im) ;
[detections, scores] = detect(im, w, hogCellSize, scales) ;


% Non-maximum suppression
keep = boxsuppress(detections, scores, 0.25) ;

detections = detections(:, keep) ;
scores = scores(keep) ;

% Further keep only top detections
umbral=5;
detections = detections(:,scores>umbral) ;
scores=scores(scores>umbral)/(max(scores)+1);

% 
% imNam=jusNam(i);
% num=length(scores);
% detC=[round(detections(1,:)'),round(detections(2,:)'),round(abs(detections(1,:)-detections(3,:))'),round(abs(detections(2,:)-detections(4,:))'),scores'];
% 
% filPath=fullfile('Results', strcat(imNam{1}(1:end-4),'.txt'));
% 
% fid=fopen(filPath,'w');
% fprintf(fid, [ imNam{1}(1:end-4) '\n']);
% fprintf(fid,'%i \n' , num);
% fprintf(fid,'%i %i %i %i %f\n' , detC');
% fclose(fid);
% 
% i/numel(trainImages)
%  end
figure(9) ; clf ;
imagesc(im) ; axis equal off ; hold on ;
vl_plotbox(detections, 'g', 'linewidth', 2) ;
title('SVM detector output') ;



