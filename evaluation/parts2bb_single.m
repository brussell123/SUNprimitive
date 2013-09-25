function bbParts = parts2bb_single(boxesParts)


xs = (boxesParts(:,1:4:end-2) + boxesParts(:,3:4:end-2))/2;
ys = (boxesParts(:,2:4:end-2) + boxesParts(:,4:4:end-2))/2;
bbParts(:,1) = min(xs,[],2);
bbParts(:,2) = min(ys,[],2);
bbParts(:,3) = max(xs,[],2);
bbParts(:,4) = max(ys,[],2);
bbParts(:,5) = boxesParts(:,end-1);
bbParts(:,6) = boxesParts(:,end);

