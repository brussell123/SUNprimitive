function edgeMap  = computeEdgeMap(im)

if size(im,3)>1
    im = rgb2gray(im);
end

edgeMap = double(edge(im,'canny',0.1, sqrt(2)* 0.005* sqrt(size(im,1) * size(im,2))));
%edgeMap = double(edge(rgb2gray(im),'canny'));

edgeMap = double(bwdist(edgeMap));

%edgeMap = edgeMap^0.5;

maxDist = (size(im,1)*size(im,2) * 0.05^2)^0.5;
edgeMap = min(edgeMap,maxDist) / maxDist;
edgeMap = imresize(edgeMap,0.25);
edgeMap = - edgeMap  * 100;


%{
%debug script;

i=1;


im = imread(testRoot(i).im);
subplot(1,6,1);
imshow(im);

cnt = 1;



subplot(1,6,cnt+1);
edge = computeEdgeMap(im);
imagesc(edge);
axis equal
axis tight
cnt = cnt + 1;
if cnt > 5
    cnt = 1;
end


target = posRoot;
indices =randperm(length(target));
for j=1:floor(length(target)/(visH*visW/2))
    maxfig(figure,1)
    visH = 5;
    visW = 10;
    cnt = 1;
    for i=indices((visH*visW/2)*(j-1)+(1:(visH*visW/2)))
        im = imread(target(i).im);
        subaxis(visH,visW, cnt, 'Spacing', 0.001, 'Padding', 0, 'Margin', 0);
        imshow(im);
        cnt=cnt+1;
        subaxis(visH,visW, cnt, 'Spacing', 0.001, 'Padding', 0, 'Margin', 0);
        edge = computeEdgeMap(im);
        imagesc(edge);
        axis equal
        axis tight
        axis off
        cnt=cnt+1;
    end
    set(gcf,'PaperPositionMode','auto')
    print('-dpng','-r200',sprintf('/csail/vision-torralba5/people/web/www/geon/canny/edge_%.3d.png',j));
    close all
end

%}