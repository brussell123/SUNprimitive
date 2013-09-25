function [boxes,model,loss,ex,mixes,exs] = detect(im, model, thresh, bbox, overlap, id, label, loss, partIDwrong, partIDcorrect)
%        [boxes,model,loss,ex] = detect(im, model, thresh, bbox, overlap, id, label, loss)
% Detect objects in image using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part
%
% If bbox is not empty, we pick best detection with significant overlap.
% If label is included, we write feature vectors to a global QP structure
%
% This function updates the model (by running the QP solver) if a large hinge loss is accumulated


% neighborhood of search
dxy = [ ...
    0 -1;
    1  0;
    0  1;
    -1  0;
    -1 -1;
    1 -1;
    1  1;
    -1  1];
dxyLength = 4;

global mineHarderIDthreshold;
global mineNegParallel;

if isempty(mineNegParallel)
    mineNegParallel = false;
end

exs={};

INF = 1e10;

if nargin>=9
    mineNegFromPos = true;
else
    mineNegFromPos = false;
end
overlapWrong = 0.2;

if nargin<10
    partIDcorrect = 1;
end

if nargin ==5 && ~isempty(bbox)
    latent = true;
    thresh = -1e10;
else
    if nargin > 3 && ~isempty(bbox) && label == 1
        latent = true;
        thresh = -1e10;
    else
        latent = false;
    end
end

% Compute the feature pyramid and prepare filter
pyra     = featpyramid(im,model);
edgeMap  = computeEdgeMap(im);

Yim2edge = (size(edgeMap,1)-1)/(pyra.imy-1);
Xim2edge = (size(edgeMap,2)-1)/(pyra.imx-1);

interval = model.interval;
levels   = 1:length(pyra.feat);

imx_half = pyra.imx/2;
imy_half = pyra.imy/2;
ims_half = sqrt(imx_half*imy_half);

% Define global QP if we are writing features
% Randomize order to increase effectiveness of model updating
write = false;
if nargin > 5,
    global qp;
    write  = true;
    levels = levels(randperm(length(levels)));
end
if nargin < 6
    id = 0;
end
if nargin < 7
    label = 0;
end
if nargin < 8
    loss = 0;
end


adjustmentEnableAll = ( ~(write || latent)  || (id<=mineHarderIDthreshold && ~mineNegFromPos) );



% Cache various statistics derived from model
[components,filters,resp] = modelcomponents(model,pyra);
boxes     = zeros(50000,length(components{1})*4+2);
mixes     = zeros(50000,length(components{1}));
ex.blocks = [];
ex.id     = [label id 0 0 0];
cnt       = 0;

% Iterate over random permutation of scales and components,
for rlevel = levels,
    % Iterate through mixture components
    for c  = randperm(length(model.components)),
        parts    = components{c};
        numparts = length(parts);
        
        iTryTotal = (numparts-1)*(dxyLength+1);
        iTryThreshold = (numparts-1)*dxyLength;
        
        
        % Skip if there is no overlap of filter with bbox
        if latent,
            skipflag = 0;
            for k = 1:numparts
                level = rlevel-parts(k).scale*interval;
                if level<1 || level > length(pyra.feat)
                    skipflag = 1;
                    break;
                end
                % because all mixtures for one part is the same size, we only need to do this once
                ovmask = testoverlap(parts(k).sizx(1),parts(k).sizy(1),pyra,level,bbox{k},overlap);
                %ovmask = testoverlap(parts(k).sizx(1),parts(k).sizy(1),pyra,rlevel,bbox{k},overlap);
                if ~any(ovmask)
                    skipflag = 1;
                    break;
                %else
                %    disp('sth');
                end
            end
            if skipflag == 1
                continue;
            end
        end
        
        
        if mineNegFromPos % to speed things up
            skipflag = 0;
            for k = 1:numparts
                level = rlevel-parts(k).scale*interval;
                if level<1 || level > length(pyra.feat)
                    skipflag = 1;
                    break;
                end
            end
            if skipflag == 1
                continue;
            end            
            for k = [partIDcorrect partIDwrong] % root must correct and partIDwrong must be wrong
                % because all mixtures for one part is the same size, we only need to do this once
                ovmask = testoverlap(parts(k).sizx(1),parts(k).sizy(1),pyra,level,bbox{k},overlap);
                %ovmask = testoverlap(parts(k).sizx(1),parts(k).sizy(1),pyra,rlevel,bbox{k},overlap);
                if k== partIDwrong
                    ovmask = ~ovmask;
                end
                if ~any(ovmask)
                    skipflag = 1;
                    break;
                %else
                %    disp('sth');
                end
            end
            if skipflag == 1
                continue;
            end
        end        
        
        
        % Local scores
        skipflag = 0;
        for k = 1:numparts,
            f     = parts(k).filterid;
            level = rlevel-parts(k).scale*interval;
            if level<1 || level > length(pyra.feat)
                skipflag = 1;
                break;
            end
            if isempty(resp{level}),
                resp{level} = fconv(pyra.feat{level},filters,1,length(filters));
            end
            for fi = 1:length(f)
                parts(k).score(:,:,fi) = resp{level}{f(fi)};
            end
            parts(k).level = level;
            
            if latent
                for fi = 1:length(f)
                    ovmask = testoverlap(parts(k).sizx(fi),parts(k).sizy(fi),pyra,level,bbox{k},overlap);
                    tmpscore = parts(k).score(:,:,fi);
                    tmpscore(~ovmask) = -INF;
                    parts(k).score(:,:,fi) = tmpscore;
                end
            end
            
            if mineNegFromPos
                if k==partIDcorrect % partIDcorrect must be correct
                    for fi = 1:length(f)
                        ovmask = testoverlap(parts(k).sizx(fi),parts(k).sizy(fi),pyra,level,bbox{k},overlap);
                        tmpscore = parts(k).score(:,:,fi);
                        tmpscore(~ovmask) = -INF;
                        parts(k).score(:,:,fi) = tmpscore;
                    end
                elseif k== partIDwrong % partIDwrong must be wrong
                    for fi = 1:length(f)
                        ovmask = testoverlap(parts(k).sizx(fi),parts(k).sizy(fi),pyra,level,bbox{k},overlapWrong);
                        ovmask = ~ovmask;
                        tmpscore = parts(k).score(:,:,fi);
                        tmpscore(~ovmask) = -INF;
                        parts(k).score(:,:,fi) = tmpscore;
                    end
                end % doesn't matters for other parts
            end
        end
        if skipflag == 1
            continue;
        end
        
        %{
        %debug test
        if numparts==4
            for p=2:numparts
                parts(p).w = rand(size(parts(p).w));
                parts(p).b = rand(size(parts(p).b));
            end
        end
        %}
        if ~(isempty(model.edge) && isempty(model.reprojection) && isempty(model.deformation))
            partsOld = parts;
        end
        
        % Walk from leaves to root of tree, passing message to parent
        for k = numparts:-1:2,
            par = parts(k).parent;
            [msg,parts(k).Ix,parts(k).Iy,parts(k).Ik] = passmsg(parts(k),parts(par));
            parts(par).score = parts(par).score + msg;
        end
        
        % Add bias to root score
        parts(1).score = parts(1).score + parts(1).b;
        [rscore Ik]    = max(parts(1).score,[],3);
        
        if isempty(model.edge) && isempty(model.reprojection) && isempty(model.deformation)
            % Zero-out invalid regions in latent mode
            if latent,
                thresh = max(thresh,max(rscore(:)));
            end
            
            [Y,X] = find(rscore >= thresh);
            
            % Walk back down tree following pointers
            % (DEBUG) Assert extracted feature re-produces score
            for i = 1:length(X)
                x = X(i);
                y = Y(i);
                k = Ik(y,x);
                [box,ex, ptr] = backtrack( x , y , k, parts , pyra , ex , latent || write);
                
                cnt = cnt + 1;
                boxes(cnt,:) = [box c rscore(y,x)];
                %boxes(cnt,:) = [box rlevel rscore(y,x)];
                mixes(cnt,:) = ptr(:,3)';
                if write && ~latent && sum(sum(isnan(ptr)))==0 % this means that only when all parts are visible, then we use it as positive traning example
                    if mineNegParallel
                        exs{end+1} = ex;
                    else
                        qp_write(ex);
                        loss  = loss + qp.Cneg*max(1+rscore(y,x),0);
                        
                        % Crucial DEBUG assertion:
                        % If we're computing features, assert extracted feature re-produces score
                        % (see qp_writ.m for computing original score)
                        if qp.n < length(qp.a)
                            w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
                            %[score(w,qp.x,qp.n), rscore(y,x), abs(score(w,qp.x,qp.n) - rscore(y,x))]
                            assert(abs(score(w,qp.x,qp.n) - rscore(y,x)) < 1e-5);
                        end
                    end
                end
            end
        else
                       
            if ~isempty(model.edge)
                edgeW = model.edge.w;
            else
                edgeW = zeros(1,size(model.index4draw,1));
            end
            
            if ~isempty(model.reprojection)
                reprojectionW = model.reprojection.w;
            else
                reprojectionW = zeros(1,7);
            end            
            
            if ~isempty(model.deformation)
                deformationW = model.deformation.w;
                
                anchorRoot(1,1,1) = 1; anchorRoot(1,1,2) = 0;
                anchorRoot(1,2,1) = 1; anchorRoot(1,2,2) = 0;
                
                for p=2:numparts
                    par = parts(p).parent;
                    anchorRoot(p,1,1) = anchorRoot(par,1,1) *partsOld(p).step;      anchorRoot(p,1,2) = (anchorRoot(par,1,2)-1) *partsOld(p).step + partsOld(p).startx(1);
                    anchorRoot(p,2,1) = anchorRoot(par,2,1) *partsOld(p).step;      anchorRoot(p,2,2) = (anchorRoot(par,2,2)-1) *partsOld(p).step + partsOld(p).starty(1);
                end
                
                upperIx = length(deformationW)/4; %+1  for debugging
                
                anchorMatrix = zeros(upperIx,2,2);
                
                for i=1:upperIx
                    par = model.index4draw(i,1);
                    chi = model.index4draw(i,2);
                    anchorMatrix(i,1,1)=anchorRoot(chi,1,1)/anchorRoot(par,1,1);    anchorMatrix(i,1,2)= anchorRoot(chi,1,2) - anchorRoot(par,1,2) * anchorRoot(chi,1,1) / anchorRoot(par,1,1) ;
                    anchorMatrix(i,2,1)=anchorRoot(chi,2,1)/anchorRoot(par,2,1);    anchorMatrix(i,2,2)= anchorRoot(chi,2,2) - anchorRoot(par,2,2) * anchorRoot(chi,2,1) / anchorRoot(par,2,1) ;
                end
                
            else
                deformationW = zeros(1,3*4);
                anchorMatrix = zeros(size(model.index4draw,1),2,2);
            end            

            if adjustmentEnableAll
                adjustThreshold = -0.5;
                rs = rscore(:);
                rsI = rs>adjustThreshold;
                if sum(rsI)>40
                    rs = rs(rsI);
                    rs = sort(rs,'descend');
                    adjustThreshold = rs(40);
                end
                %fprintf('%d\n', sum(sum(rscore>=adjustThreshold)));
                %{
                rs = sort(rscore(:),'descend');
                adjustThreshold = min(100,ceil(length(rs)*0.10));
                adjustThreshold = max(-0.8,rs(adjustThreshold));
                fprintf('%f\n',adjustThreshold);
                %}
            end
            
            if latent
                %[Y,X] = find(rscore >= max(rscore(:)));
                [Y,X] = find(rscore >= max(max(rscore(:)),-INF+1)); % for latent positive mining, if no solution at all. skip
            else
                [Y,X] = find(rscore >= thresh);
            end
            
            
            
            for i = 1:length(X)
                x = X(i);
                y = Y(i);
                [box,junk, ptr] = backtrack( x , y , Ik(y,x), parts , pyra , ex , false);
                
                % we can randomly choose a small subset to do this during training. 
                adjustmentEnable = (~any(isnan(ptr(:)))) && adjustmentEnableAll && (rscore(y,x)>=adjustThreshold);

                
                updateIndex = 1:numparts;
                
                ixyCurrent = ones(numparts,3);
                pxyCurrent = zeros(numparts,2);
                
                appearanceValuesCurrent = zeros(1,numparts);
                
                deformationValuesCurrent = zeros(1,numparts+(length(deformationW)/4));
                vec_deformation = zeros(1,length(deformationW));
                
                edgeValuesCurrent = zeros(1,length(edgeW));
                vec_edge = zeros(1,length(edgeW));
                
                vec_reprojection = zeros(1,numparts);
                
                [scoreMax, ixyCurrent, pxyCurrent, appearanceValuesCurrent, deformationValuesCurrent, vec_deformation, edgeValuesCurrent, vec_edge, vec_reprojection] = ...
                    objectiveFunction(  ptr, partsOld, pyra.padx, pyra.pady, pyra.scale, imx_half, imy_half, ims_half, ...
                    updateIndex, ixyCurrent, pxyCurrent, Xim2edge, Yim2edge, ...
                    appearanceValuesCurrent, ...
                    deformationW, vec_deformation, deformationValuesCurrent, anchorMatrix, ...
                    edgeW, vec_edge, edgeValuesCurrent, edgeMap, model.index4draw, ...
                    reprojectionW, vec_reprojection);


                    
                % antonio's idea: maybe disable this during training
                % <= it just mean that we cannot find very very hard negatives
                % <= and we don't mine positve much
                if adjustmentEnable 
                    % optimize to look for a better location for all parts and scores
                    % by gradiant descent or local search
                    % ptr  <= new ptr to look for better score
                    
                    iterationCnt = 0;
                    clear ptrA;
                    clear ptrC;
                    
                    foundBetter = true;
                    %ptrQueue = ptr; %<= to handle uniqueness
                    
                    %tic
                    
                    while foundBetter
                        foundBetter = false;
                        
                        %for iTry = [(iTryThreshold+randperm(numparts-1)) randperm(iTryThreshold)]
                        for iTry = randperm(iTryTotal) %1:iTryTotal %
                            
                            
                            % get the new ptr for the chosenCorner
                            if iTry <= iTryThreshold
                                newCornerID = 1+ceil(iTry/dxyLength);
                                idxy = 1+mod(iTry,dxyLength);
                                ptrNew = ptr;
                                ptrNew(newCornerID,1:2) = ptr(newCornerID,1:2) + dxy(idxy,:);
                            else
                                if size(ixyCurrent,1)~=7
                                    continue;
                                end
                                
                                % Geometric adjustment
                                newCornerID = iTry-((numparts-1)*dxyLength)+1;

                                
                                %ptrNew(newCornerID,1:2) <= new corners here
                                ixyNew = newCorner(ixyCurrent, newCornerID);
                                
                                ptrNew=ixy2ptr(ixyNew, ptr, partsOld, pyra.padx, pyra.pady, pyra.scale, imx_half, imy_half, ims_half, newCornerID);
                                
                                if sum(ptrNew(newCornerID,1:2) == ptr(newCornerID,1:2))==2 % it means nothing changed.
                                    continue;
                                end
                            end
                            
                            iterationCnt = iterationCnt + 1;
                            ptrA{iterationCnt}=ptrNew;
                            ptrC{iterationCnt}=ptr;
                            
                            % see if the new solution is validate at all
                            if isnan(ptrNew(newCornerID,1)) || isnan(ptrNew(newCornerID,2)) || ptrNew(newCornerID,1)<1 || ptrNew(newCornerID,1) > size(partsOld(newCornerID).score,2) || ptrNew(newCornerID,2)<1 || ptrNew(newCornerID,2) > size(partsOld(newCornerID).score,1) 
                                continue;
                            end

                            % get the new appearance objective function
                            appearanceValuesNew = appearanceScore(ptrNew,partsOld,appearanceValuesCurrent,newCornerID);
                            scoreNew = sum(appearanceValuesNew);
                            
                            % if it is better than scoreMax, go ahead
                            % apearance may not be the upper bound, so it
                            % is more correct to not just skip from here
                            %if scoreNew < scoreMax
                            %    continue;
                            %end
                            
                            
                            % get the new 2D deformation score
                            [deformationValuesNew, vec_deformationNew] = deformationScore(ptrNew, partsOld, deformationW, vec_deformation, deformationValuesCurrent, anchorMatrix, model.index4draw, newCornerID);
                            scoreNew = scoreNew + sum(deformationValuesNew);
                            
                            % if it is better than scoreMax, go ahead
                            if scoreNew < scoreMax
                                continue;
                            end
                            
                            % see if the new solution has been examed before
                            % <= if examing the uniqueness has longer time
                            % than just try it, we should remove this
                            % ptrQueue
                            [ixyNew, pxyNew] = ptr2ixy(ptrNew, partsOld, pyra.padx, pyra.pady, pyra.scale, imx_half, imy_half, ims_half, newCornerID, ixyCurrent, pxyCurrent, Xim2edge, Yim2edge);
                            
                            
                            % reprojection errors
                            [reprojectionValue, vec_reprojectionNew] = reprojectionScore(reprojectionW, ixyNew);
                            scoreNew = scoreNew + reprojectionValue;
                            if scoreNew < scoreMax
                                continue;
                            end                            

                            
                            % get the new edge score
                            [edgeValuesNew, vec_edgeNew] = edgeScore(pxyNew, edgeW, vec_edge, edgeValuesCurrent, edgeMap, model.index4draw, newCornerID);
                            scoreNew = scoreNew + sum(edgeValuesNew);
                            
                            
                            %iterationCnt = iterationCnt+1;
                            
                            % if it is better than scoreMax, we got a better solution!
                            if scoreNew > scoreMax
                                
                                %fprintf('%f => %f \n', scoreMax, scoreNew);
                                
                                scoreMax = scoreNew;
                                foundBetter = true;

                                %ptrQueue(:,:,end+1) = ptrNew;

                                appearanceValuesCurrent = appearanceValuesNew;
                                
                                deformationValuesCurrent = deformationValuesNew;
                                vec_deformation = vec_deformationNew;
                                
                                ptr = ptrNew;
                                ixyCurrent = ixyNew;
                                pxyCurrent = pxyNew;
                                
                                
                                edgeValuesCurrent = edgeValuesNew;
                                vec_edge = vec_edgeNew;
                                
                                vec_reprojection = vec_reprojectionNew;
                                
                                % visualization of animation for refinement
                                % <= make a movie here in supplementary?
                                

                                % if break: it is simple hill climbing
                                % if no break: it is steepest ascent hill climbing                                
                                break; 
                            end
                        end
                    end
                    %fprintf('%d\n', iterationCnt);
                    %toc
                end
                
                rscore(y,x) = scoreMax;
                
                % Zero-out invalid regions in latent mode
                if rscore(y,x) > thresh
                    
                    if adjustmentEnable
                        % get the actual image box from ptr
                        % only necessary if the parts are adjusted
                        % otherwise the backtrace already handles that
                        % box <= ptr;
                        clear box
                        for p=1:numparts
                            scale = pyra.scale(partsOld(p).level);
                            mix = ptr(p,3);
                            if isnan(mix)
                                box(p,:) = [NaN NaN NaN NaN];
                            else
                                box(p,:) = [(ptr(p,1) - 1 - pyra.padx)*scale+1, (ptr(p,2) - 1 - pyra.pady)*scale+1, (ptr(p,1) - 1 - pyra.padx)*scale + parts(p).sizx(mix)*scale, (ptr(p,2) - 1 - pyra.pady)*scale + parts(p).sizy(mix)*scale ];
                            end
                        end
                        box = reshape(box',1,4*numparts);
                    end
                    
                    cnt = cnt + 1;
                    boxes(cnt,:) = [box c rscore(y,x)];
                    %boxes(cnt,:) = [box i rscore(y,x)];
                    mixes(cnt,:) = ptr(:,3);
                    
                    % get the actual HOG feature + deformation feature + shape feature from ptr
                    % ex <= HOG(ptr);
                    if (write || latent) && sum(isnan(ptr(:,3)))==0
                        ex.id(3:5) = [partsOld(1).level round(ptr(1,1)+partsOld(1).sizx(ptr(1,3))/2) round(ptr(1,2)+partsOld(1).sizy(ptr(1,3))/2)];
                        ex.blocks = [];
                        
                        if ~isempty(model.edge)
                            ex.blocks(end+1).i = model.edge.i;
                            ex.blocks(end).x   = vec_edge;
                        end
                        
                        if ~isempty(model.reprojection)
                            ex.blocks(end+1).i = model.reprojection.i;
                            ex.blocks(end).x   = vec_reprojection;
                        end           
                        
                        if ~isempty(model.deformation)
                            ex.blocks(end+1).i = model.deformation.i;
                            ex.blocks(end).x   = vec_deformation;
                        end                                
                        
                        
                        ex.blocks(end+1).i = partsOld(1).biasI;
                        ex.blocks(end).x   = 1;
                        f  = pyra.feat{partsOld(1).level}(ptr(1,2):ptr(1,2)+partsOld(1).sizy(ptr(1,3))-1,ptr(1,1):ptr(1,1)+partsOld(1).sizx(ptr(1,3))-1,:);
                        ex.blocks(end+1).i = partsOld(1).filterI(ptr(1,3));
                        ex.blocks(end).x   = f;
                        for k=2:numparts
                            p = partsOld(k);
                            par = p.parent;
                            mix = ptr(par,3);
                            ex.blocks(end+1).i = p.biasI(mix,ptr(k,3));
                            ex.blocks(end).x   = 1;
                            ex.blocks(end+1).i = p.defI(ptr(k,3));
                            ex.blocks(end).x   = defvector(ptr(par,1),ptr(par,2),ptr(k,1),ptr(k,2),ptr(k,3),p);
                            mix = ptr(k,3);
                            if 0<mix && 0<ptr(k,2) && ptr(k,2)+p.sizy(mix)-1 <= size(pyra.feat{p.level},1) && 0<ptr(k,1) && ptr(k,1)+p.sizx(mix)-1 <= size(pyra.feat{p.level},2)
                                f   = pyra.feat{p.level}(ptr(k,2):ptr(k,2)+p.sizy(mix)-1,ptr(k,1):ptr(k,1)+p.sizx(mix)-1,:);
                            else
                                f   = zeros(p.sizy(mix),p.sizx(mix),size(pyra.feat{p.level},3));
                                fprintf('\nPart %d is outside the image.\n', k);
                            end
                            ex.blocks(end+1).i = p.filterI(mix);
                            ex.blocks(end).x = f;
                        end
                    end
                    
                    if write && ~latent && sum(isnan(ptr(:,3)))==0 % this means that only when all parts are visible, then we use it as negative traning example
                        if mineNegParallel
                            exs{end+1} = ex;
                        else
                            qp_write(ex);
                            loss  = loss + qp.Cneg*max(1+rscore(y,x),0);
                            
                            % Crucial DEBUG assertion:
                            % If we're computing features, assert extracted feature re-produces score
                            % (see qp_writ.m for computing original score)
                            %{
                            if qp.n < length(qp.a)
                                w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
                                %[score(w,qp.x,qp.n), rscore(y,x), abs(score(w,qp.x,qp.n) - rscore(y,x))]
                                if abs(score(w,qp.x,qp.n) - rscore(y,x)) >= 1e-5
                                    fprintf('\n assert: %f %f %f\n', score(w,qp.x,qp.n), rscore(y,x), abs(score(w,qp.x,qp.n) - rscore(y,x)));
                                end
                                assert(abs(score(w,qp.x,qp.n) - rscore(y,x)) < 1e-5);
                            end
                            %}
                        end
                    end
                end
            end
        end
        
        % Crucial DEBUG assertion:
        % If we're computing features, assert extracted feature re-produces score
        % (see qp_writ.m for computing original score)
        % this code is buggy, because if no (x,y) in X is larger than the
        % threshold, it will break
        %{
        if write && ~latent && ~isempty(X) && qp.n < length(qp.a),
            w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
            %w = qp_w;
            assert(abs(score(w,qp.x,qp.n) - rscore(y,x)) < 1e-5);
        end
        %}
        
        % Optimize qp with coordinate descent, and update model
        
        if write && ~latent && ~mineNegParallel && (qp.obj < 0 || loss/qp.obj > .01 || qp.n == length(qp.sv))
            model = optimize(model);
            [components,filters,resp] = modelcomponents(model,pyra);
            loss = 0;
        end
    end
end

boxes = boxes(1:cnt,:);
mixes = mixes(1:cnt,:);

%{
imshow(im)
plot_bbox(bbox{1},'truth',[0 1 0],[1 0 1]);
for i=1:cnt
    plot_bbox(boxes(i,1:4));
end
%}


if latent && ~isempty(boxes),
    boxes = boxes(end,:);
    mixes = mixes(end,:);
    if write && sum(isnan(mixes))==0
        qp_write(ex);
    end
end

% ----------------------------------------------------------------------
% objective function
% ----------------------------------------------------------------------

function [ixy pxy]=ptr2ixy(ptr, partsOld, pyra_padx, pyra_pady, pyra_scale, imx_half, imy_half, ims_half, updateIndex, ixy, pxy, Xim2edge, Yim2edge)
for p=updateIndex
    mix = ptr(p,3);
    if isnan(mix)
        ixy(p,1:2) = [NaN NaN];
        pxy(p,1:2) = [NaN NaN];
    else
        pxy(p,1:2) = [(ptr(p,1)-0.5-pyra_padx +partsOld(p).sizx(mix)/2), (ptr(p,2)-0.5-pyra_pady +partsOld(p).sizy(mix)/2)]*pyra_scale(partsOld(p).level);
        ixy(p,1)   = pxy(p,1)   - imx_half;
        ixy(p,2)   = pxy(p,2)   - imy_half;
        ixy(p,1:2) = ixy(p,1:2) / ims_half;
        
        pxy(p,1) = pxy(p,1)*Xim2edge;
        pxy(p,2) = pxy(p,2)*Yim2edge;
    end
end

function ptr=ixy2ptr(ixy, ptr, partsOld, pyra_padx, pyra_pady, pyra_scale, imx_half, imy_half, ims_half, updateIndex)
for p=updateIndex
    mix = ptr(p,3);
    if isnan(mix)
        ptr(p,1:2) = [NaN NaN];
    else
        x = (ixy(p,1) * ims_half + imx_half)/pyra_scale(partsOld(p).level);
        y = (ixy(p,2) * ims_half + imy_half)/pyra_scale(partsOld(p).level);
        ptr(p,1) = round(x - partsOld(p).sizx(mix)/2 + pyra_padx + 0.5);
        ptr(p,2) = round(y - partsOld(p).sizy(mix)/2 + pyra_pady + 0.5);
    end
end

function values = appearanceScore(ptr,partsOld,values,updateIndex)
%numparts = length(partsOld);
%if nargin<3
%    values = zeros(1,numparts);
%end
%if nargin<4
%    updateIndex = 1:numparts;
%end
for p=updateIndex
    if p==1
        values(p) = partsOld(1).b + partsOld(1).score(ptr(1,2),ptr(1,1),ptr(1,3));
    else
        kc = ptr(p,3);
        kp = ptr(partsOld(p).parent,3);
        
        if ~isnan(kp)
            if ~isnan(kc)
                values(p) = partsOld(p).b(1,kp,kc) + partsOld(p).score(ptr(p,2),ptr(p,1),kc);
            else
                values(p) = -1e10;
            end
        else
            values(p) = 0;
        end
    end
end

function [values, vec_deformation] = deformationScore(ptr, partsOld, deformationW, vec_deformation, values, anchorMatrix, index4draw, updateIndex)
numparts = length(partsOld);

% non tree edges
for i=1:(length(deformationW)/4)
    if any(updateIndex==index4draw(i,1)) || any(updateIndex==index4draw(i,2))
        if ~isnan(ptr(index4draw(i,1),3)) && ~isnan(ptr(index4draw(i,2),3))
            idxRange = ((i-1)*4+1):(i*4);
            
            xp = ptr(index4draw(i,1),1);
            yp = ptr(index4draw(i,1),2);
            %kp = ptr(index4draw(i,1),3);
            xc = ptr(index4draw(i,2),1);
            yc = ptr(index4draw(i,2),2);
            %kc = ptr(index4draw(i,2),3);
            
            dx = xp * anchorMatrix(i,1,1)+anchorMatrix(i,1,2) - xc;
            dy = yp * anchorMatrix(i,2,1)+anchorMatrix(i,2,2) - yc;
            
            vec_deformation(idxRange) = - [dx^2 dx dy^2 dy];
            values(numparts+i) = deformationW(idxRange) * vec_deformation(idxRange)';
        else
            values(numparts+i) = 0;
        end
    end
end


%if nargin<3
%    values = zeros(1,numparts);
%end
%if nargin<4
%    updateIndex = 1:numparts;
%else
    % transform updateIndex, either parent or children change, it need to be change. parent always comes first
    for p=1:numparts
        par = partsOld(p).parent;
        if any(updateIndex==par) % if you parent change, you need to change
            updateIndex = [updateIndex p]; 
        end
    end
    updateIndex=unique(updateIndex); % remove duplicate entry
%end

for p=updateIndex
    if p==1
        values(p) = 0;
    else
        kc = ptr(p,3);
        par = partsOld(p).parent;
        kp = ptr(par,3);
        if ~isnan(kp) && ~isnan(kc)
            xc = ptr(p,1);
            yc = ptr(p,2);
            xp = ptr(par,1);
            yp = ptr(par,2);
            dx = (xp-1)*partsOld(p).step + partsOld(p).startx(kc) - xc;
            dy = (yp-1)*partsOld(p).step + partsOld(p).starty(kc) - yc;
            res = [dx^2 dx dy^2 dy];
            values(p) = - res * partsOld(p).w(:,kc);
        else
            values(p) = 0;
        end
    end
end

function [edgeValues, vec_edge] = edgeScore(pxy, edgeW, vec_edge, edgeValues, edgeMap, index4draw, updateIndex)

for i=1:length(edgeW)
    if any(updateIndex==index4draw(i,1)) || any(updateIndex==index4draw(i,2))
        [lx ly] = bresenham(pxy(index4draw(i,1),1),pxy(index4draw(i,1),2),pxy(index4draw(i,2),1),pxy(index4draw(i,2),2));
        if isempty(lx)
            vec_edge(i) = - 100;
        else
            selectIdx = lx>0 & ly>0 & lx<=size(edgeMap,2) & ly<=size(edgeMap,1);
            if any(selectIdx)
                lx = lx(selectIdx);
                ly = ly(selectIdx);
                %vec_edge(i) = - mean(edgeMap(sub2ind(size(edgeMap), ly, lx)));
                %vec_edge(i) = min(edgeMap(sub2ind(size(edgeMap), ly, lx)));
                %vec_edge(i) = median(edgeMap(sub2ind(size(edgeMap), ly, lx))); % at least 50% points must be an edge
                vec_edge(i) = mean(edgeMap(sub2ind(size(edgeMap), ly, lx))); % at least 50% points must be an edge
            else
                vec_edge(i) = - 100;
            end
        end
        edgeValues(i) = edgeW(i) * vec_edge(i);
    end
end


function ixy = newCorner(ixy, newCornerID)
numparts = 7;
idx = true(1, numparts);
idx(newCornerID) = false;
[junk,junk,ixy(newCornerID,1:2)]=reconstructCuboid(ixy,idx);


% this only for cuboid with 7 corners
function [reprojectionValue, vec_reprojection] = reprojectionScore(reprojectionW, ixy)
numparts = 7;

if sum(reprojectionW==zeros(1,7))==7
    vec_reprojection = zeros(1,7);
    reprojectionValue = 0;
else
    vec_reprojection = zeros(1,numparts);
    for p=1:numparts
        idx = true(1, numparts);
        idx(p) = false;
        vec_reprojection(p) = - reconstructCuboid(ixy,idx);
    end
    reprojectionValue = reprojectionW * vec_reprojection';
end


function [new_score, ixy,  pxy, appearanceValues, deformationValues, vec_deformation, edgeValues, vec_edge, vec_reprojection] = ...
    objectiveFunction(  ptr, partsOld, pyra_padx, pyra_pady, pyra_scale, imx_half, imy_half, ims_half, ...
                        updateIndex, ixy, pxy, Xim2edge, Yim2edge, ...
                        appearanceValues, ...
                        deformationW, vec_deformation, deformationValues, anchorMatrix, ...
                        edgeW, vec_edge, edgeValues, edgeMap, index4draw, ...
                        reprojectionW, vec_reprojection)


if any(isnan(ptr)) %speed up
    
    new_score = -1e10;
    
else
    appearanceValues = appearanceScore(ptr, partsOld, appearanceValues, updateIndex);
    
    [deformationValues, vec_deformation] = deformationScore(ptr, partsOld, deformationW, vec_deformation, deformationValues, anchorMatrix, index4draw, updateIndex);
    
    [ixy, pxy] = ptr2ixy(ptr, partsOld, pyra_padx, pyra_pady, pyra_scale, imx_half, imy_half, ims_half, updateIndex, ixy, pxy, Xim2edge, Yim2edge);
    
    [edgeValues, vec_edge] = edgeScore(pxy, edgeW, vec_edge, edgeValues, edgeMap, index4draw, updateIndex);
    
    [reprojectionValue, vec_reprojection] = reprojectionScore(reprojectionW, ixy);
    
    new_score = sum(appearanceValues) + sum(deformationValues) + sum(edgeValues) + reprojectionValue;
end



% ----------------------------------------------------------------------
% Helper functions for detection, feature extraction, and model updating
% ----------------------------------------------------------------------

% Cache various statistics from the model data structure for later use
function [components,filters,resp] = modelcomponents(model,pyra)
components = cell(length(model.components),1);
for c = 1:length(model.components),
    for k = 1:length(model.components{c}),
        p = model.components{c}(k);
        [p.sizy,p.sizx,p.w,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
        [p.scale,p.level,p.Ix,p.Iy] = deal(0);
        
        % store the scale of each part relative to the component root
        par = p.parent;
        assert(par < k);
        p.b = [model.bias(p.biasid).w];
        p.b = reshape(p.b,[1 size(p.biasid)]);
        p.biasI = [model.bias(p.biasid).i];
        p.biasI = reshape(p.biasI,size(p.biasid));
        
        for f = 1:length(p.filterid)
            x = model.filters(p.filterid(f));
            [p.sizy(f) p.sizx(f) foo] = size(x.w);
            p.filterI(f) = x.i;
        end
        
        for f = 1:length(p.defid)
            x = model.defs(p.defid(f));
            p.w(:,f)  = x.w';
            p.defI(f) = x.i;
            ax  = x.anchor(1);
            ay  = x.anchor(2);
            ds  = x.anchor(3);
            p.scale = ds + components{c}(par).scale;
            % amount of (virtual) padding to hallucinate
            step     = 2^ds;
            virtpady = (step-1)*pyra.pady;
            virtpadx = (step-1)*pyra.padx;
            % starting points (simulates additional padding at finer scales)
            p.starty(f) = ay-virtpady;
            p.startx(f) = ax-virtpadx;
            p.step   = step;
        end
        components{c}(k) = p;
    end
end

resp    = cell(length(pyra.feat),1);
filters = cell(length(model.filters),1);
for i = 1:length(filters),
    filters{i} = model.filters(i).w;
end

% Given a 2D array of filter scores 'child',
% (1) Apply distance transform
% (2) Shift by anchor position of part wrt parent
% (3) Downsample if necessary
function [score,Ix,Iy,Ik] = passmsg(child,parent)
INF = 1e10;
K   = length(child.filterid);
Ny  = size(parent.score,1);
Nx  = size(parent.score,2);
Ix0 = zeros([Ny Nx K]);
Iy0 = zeros([Ny Nx K]);
score0 = repmat(-INF,[Ny Nx K]);

for k = 1:K
    [score_tmp,Ix_tmp,Iy_tmp] = dt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k));
    
    % starting points
    startx = child.startx(k);
    starty = child.starty(k);
    step   = child.step;
    % ending points
    endy = starty+step*(Ny-1);
    endx = startx+step*(Nx-1);
    endy = min(size(child.score,1),endy);
    endx = min(size(child.score,2),endx);
    % y sample points
    iy = starty:step:endy;
    oy = sum(iy < 1);
    iy = iy(iy >= 1);
    % x sample points
    ix = startx:step:endx;
    ox = sum(ix < 1);
    ix = ix(ix >= 1);
    % sample scores
    sp = score_tmp(iy,ix);
    sx = Ix_tmp(iy,ix);
    sy = Iy_tmp(iy,ix);
    sz = size(sp);
    % define msgs
    iy  = oy+1:oy+sz(1);
    ix  = ox+1:ox+sz(2);
    
    score0(iy,ix,k) = sp;
    Ix0(iy,ix,k)    = sx;
    Iy0(iy,ix,k)    = sy;
end

% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
L  = length(parent.filterid);
N  = Nx*Ny;
i0 = reshape(1:N,Ny,Nx);
[score,Ix,Iy,Ix] = deal(zeros(Ny,Nx,L));
for l = 1:L
    b = child.b(1,l,:);
    [score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
    i = i0 + N*(I-1);
    Ix(:,:,l)    = Ix0(i);
    Iy(:,:,l)    = Iy0(i);
    Ik(:,:,l)    = I;
end

% Backtrack through dynamic programming messages to estimate part locations
% and the associated feature vector
function [box,ex, ptr] = backtrack(x,y,mix,parts,pyra,ex,write)

numparts = length(parts);
ptr = zeros(numparts,3);
box = zeros(numparts,4);
k   = 1;
p   = parts(k);
ptr(k,:) = [x y mix];
scale = pyra.scale(p.level);
x1  = (x - 1 - pyra.padx)*scale+1;
y1  = (y - 1 - pyra.pady)*scale+1;
x2  = x1 + p.sizx(mix)*scale - 1;
y2  = y1 + p.sizy(mix)*scale - 1;
box(k,:) = [x1 y1 x2 y2];

if write,
    ex.id(3:5) = [p.level round(x+p.sizx(mix)/2) round(y+p.sizy(mix)/2)];
    ex.blocks = [];
    ex.blocks(end+1).i = p.biasI;
    ex.blocks(end).x   = 1;
    f  = pyra.feat{p.level}(y:y+p.sizy(mix)-1,x:x+p.sizx(mix)-1,:);
    ex.blocks(end+1).i = p.filterI(mix);
    ex.blocks(end).x   = f;
end
for k = 2:numparts,
    p   = parts(k);
    par = p.parent;
    x   = ptr(par,1);
    y   = ptr(par,2);
    mix = ptr(par,3);
    if ~isnan(x) && ~isnan(y) && ~isnan(mix) && 0<y && y<=size(p.Ix,1) && 0<x && x<=size(p.Ix,2) && 0<mix && mix<=size(p.Ix,3)
        ptr(k,1) = p.Ix(y,x,mix);
        ptr(k,2) = p.Iy(y,x,mix);
        ptr(k,3) = p.Ik(y,x,mix);
        scale = pyra.scale(p.level);
        x1  = (ptr(k,1) - 1 - pyra.padx)*scale+1;
        y1  = (ptr(k,2) - 1 - pyra.pady)*scale+1;
        x2  = x1 + p.sizx(ptr(k,3))*scale - 1;
        y2  = y1 + p.sizy(ptr(k,3))*scale - 1;
        box(k,:) = [x1 y1 x2 y2];
        if write
            ex.blocks(end+1).i = p.biasI(mix,ptr(k,3));
            ex.blocks(end).x   = 1;
            ex.blocks(end+1).i = p.defI(ptr(k,3));
            ex.blocks(end).x   = defvector(x,y,ptr(k,1),ptr(k,2),ptr(k,3),p);
        end
        x   = ptr(k,1);
        y   = ptr(k,2);
        mix = ptr(k,3);
        if 0<mix && 0<y && y+p.sizy(mix)-1 <= size(pyra.feat{p.level},1) && 0<x && x+p.sizx(mix)-1 <= size(pyra.feat{p.level},2)
            if write
                f   = pyra.feat{p.level}(y:y+p.sizy(mix)-1,x:x+p.sizx(mix)-1,:);
            end
        else
            if write
                f   = zeros(p.sizy(mix),p.sizx(mix),size(pyra.feat{p.level},3));
            end
            ptr(k,:) = [NaN NaN NaN];
            box(k,:) = [NaN NaN NaN NaN];
            %fprintf('\nPart %d is outside the image.\n', k);
        end
        if write
            ex.blocks(end+1).i = p.filterI(mix);
            ex.blocks(end).x = f;
        end
    else
        ptr(k,1) = NaN;
        ptr(k,2) = NaN;
        ptr(k,3) = NaN;
        box(k,:) = [NaN NaN NaN NaN];
        %fprintf('\nParent %d of Part %d is outside the image.\n', par, k);
    end
end
box = reshape(box',1,4*numparts);

% Update QP with coordinate descent
% and return the asociated model
function model = optimize(model)
global qp;
fprintf('.');
if qp.obj < 0 || qp.n == length(qp.a),
    qp_opt(1);
    qp_prune();
else
    qp_one();
end
model = vec2model(qp_w,model);

% Compute the deformation feature given parent locations,
% child locations, and the child part
function res = defvector(px,py,x,y,mix,part)
probex = ( (px-1)*part.step + part.startx(mix) );
probey = ( (py-1)*part.step + part.starty(mix) );
dx  = probex - x;
dy  = probey - y;
res = -[dx^2 dx dy^2 dy]';

% Compute a mask of filter reponse locations (for a filter of size sizy,sizx)
% that sufficiently overlap a ground-truth bounding box (bbox)
% at a particular level in a feature pyramid
function ov = testoverlap(sizx,sizy,pyra,level,bbox,overlap)
scale = pyra.scale(level);
padx  = pyra.padx;
pady  = pyra.pady;
[dimy,dimx,foo] = size(pyra.feat{level});

bx1 = bbox(1);
by1 = bbox(2);
bx2 = bbox(3);
by2 = bbox(4);

% Index windows evaluated by filter (in image coordinates)
x1 = ((1:dimx-sizx+1) - padx - 1)*scale + 1;
y1 = ((1:dimy-sizy+1) - pady - 1)*scale + 1;
x2 = x1 + sizx*scale - 1;
y2 = y1 + sizy*scale - 1;

% Compute intersection with bbox
xx1 = max(x1,bx1);
xx2 = min(x2,bx2);
yy1 = max(y1,by1);
yy2 = min(y2,by2);
w   = xx2 - xx1 + 1;
h   = yy2 - yy1 + 1;
w(w<0) = 0;
h(h<0) = 0;
inter  = h'*w;

% area of (possibly clipped) detection windows and original bbox
area = (y2-y1+1)'*(x2-x1+1);
box  = (by2-by1+1)*(bx2-bx1+1);

% thresholded overlap
ov   = inter ./ (area + box - inter) > overlap;




