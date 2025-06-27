function [judge] = judge_point(x,y,xmlout)
% JUDGE_POINT Determines if a point is inside a closed polygon region
%   This function checks whether the given (x,y) coordinates lie inside
%   any polygon regions defined in an XML structure
%
% Inputs:
%   x, y  : Coordinates of the point to test
%   xmlout: Parsed XML structure containing polygon regions
%
% Output:
%   judge : Returns 1 if point is inside any polygon, 0 otherwise

count=0;
for j = 2:2:length(xmlout.Children(2).Children(8).Children)
    mid1 = xmlout.Children(2).Children(8).Children(j).Children;
    ind = 0;  
    mid2 = mid1(6).Children;
    ind = ind + 1;
    x_total = [];
    y_total = [];
    ind1 = 0; 
    for m = 2:2:length(mid2)
        temp=0;
        ind1 = ind1 + 1;
        temp=split(mid2(m).Children.Data,',');
        x_total(ind1, 1) = floor(str2double(temp(1)));
        y_total(ind1, 1) = floor(str2double(temp(2)));
    end
    x_total(ind1+1)=x_total(1);
    y_total(ind1+1)=y_total(1);
    for i=1:(length(y_total)-1)
        if (x_total(i)>=x||x_total(i+1)>=x)&&((y_total(i)>=y&&y_total(i+1)<=y)||(y_total(i)<=y&&y_total(i+1)>=y))
            count=count+1;
        end
    end
    if  mod(count,2)==1
        judge=1;
    else
        judge=0;

    end
end


