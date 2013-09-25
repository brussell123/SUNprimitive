% compile.m <- run for the first time

globals;

% Load trained cuboid model:
load('model_cuboid.mat');

% Load example image:
img = imread('example.jpg');

% Run cuboid detector:
out = DetectCuboids(img,model);

% Display detected cuboids:
figure;
imshow(img);
hold on;
for j = 1:size(out.boxes,1)
  if out.boxes(j,end) > -0.1
    showshapeUniqueColorSolidLine(out.boxes(j,:),model.index4draw,'g');
  end
end
