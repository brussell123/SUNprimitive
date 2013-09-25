function out = DetectCuboids(img,model)

globals;

numparts = length(model.filters);

% Initialize outputs:
out = struct('boxes',[],'points',[],'mixes',[]);

% Run detector:
warning('off','MATLAB:rankDeficientMatrix');
[box,junk,junk,junk,mix] = detect(img,model,model.thresh);    
[boxHflip,junk,junk,junk,mixHflip] = detect(img(:,end:-1:1,:),model,model.thresh);    
warning('on','MATLAB:rankDeficientMatrix');

% our overall nms
if ~isempty(box)
  [box pick] = nms(box,0.5);
  mix = mix(pick,:);
end
if ~isempty(boxHflip)
  [boxHflip pickHflip] = nms(boxHflip,0.5);
  mixHflip = mixHflip(pickHflip,:);

  % Flip outputs back to correct direction:
  x2 = size(img,2)+1-boxHflip(:,1:4:(numparts*4));
  x1 = size(img,2)+1-boxHflip(:,3:4:(numparts*4));
  boxHflip(:,1:4:(numparts*4))=x1;
  boxHflip(:,3:4:(numparts*4))=x2;
end

% Combine both sets of detections:
box = [box; boxHflip];
mix = [mix; mixHflip];

if ~isempty(box)
  bb = parts2bb_single(box);
  [junk,pick] = nms(bb,0.5);
  out.boxes = box(pick,:);
  out.mixes = mix(pick,:);
  for p = 1:numparts
    cx1 = out.boxes(:,1+(p-1)*4);
    cy1 = out.boxes(:,2+(p-1)*4);
    cx2 = out.boxes(:,3+(p-1)*4);
    cy2 = out.boxes(:,4+(p-1)*4);
    out.points(:,1+(p-1)*2) = (cx1+cx2)/2;
    out.points(:,2+(p-1)*2) = (cy1+cy2)/2;
  end
end
