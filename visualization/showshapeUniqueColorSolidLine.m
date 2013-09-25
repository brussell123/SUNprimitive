function showshapeUniqueColorSolidLine(boxes, index4draw, color, numparts)

if nargin<3
    color = 'b';
end

if nargin<4
    numparts = 7;
end

hold on;
if ~isempty(boxes)
    if size(boxes,2)==2*numparts
        x = boxes(:,1:2:end);
        y = boxes(:,2:2:end);
        boxes(:,end+2)=0;
    else
        for p = 1:floor(size(boxes,2)/4)
            x1(:,p) = boxes(:,1+(p-1)*4);
            y1(:,p) = boxes(:,2+(p-1)*4);
            x2(:,p) = boxes(:,3+(p-1)*4);
            y2(:,p) = boxes(:,4+(p-1)*4);
        end
        x = (x1 + x2)/2;
        y = (y1 + y2)/2;
    end
    
    for n = 1:size(boxes,1)
        
        score = boxes(n,end);
        score = score + 2;   
        if score<=0
            score = 0.00001;
        end
        
        score = score * 1.2;
        
        for i=1:size(index4draw,1)
            x1 = x(n,index4draw(i,1));
            y1 = y(n,index4draw(i,1));
            x2 = x(n,index4draw(i,2));
            y2 = y(n,index4draw(i,2));
            

                colorNow=color;

            

                LineStyle = '-';

            
            if score==inf
                linewdith = 3;
            else
                linewdith = score;
            end
            
            line([x1 x2],[y1 y2],'LineStyle',LineStyle,'color',colorNow,'linewidth',linewdith);
                
        end
    end
end
drawnow;
