clc
clear
close all

openslide_load_library();

addpath(genpath('./subroutines/'));
mkdir('./output_images');
mkdir('./output_masks');
mkdir('./output_rgbout');
load('./model/model1.mat');

% Define color map for tissue classes in RGB format (Gray, Cyan, Orange, Blue, Magenta, Yellow, Green, Red) 
colors = [ 192, 192, 192; 0, 255, 255; 255, 165, 0; 0, 0, 255; 255, 0, 255; 255, 255, 0; ...
    0, 255, 0; 255, 0, 0]/255;

tissuenames = {'ADI','BACK','INT','LYM','MUS','NORM','STR','TUM'};
border = [76 76];% Overlap border between patches

% Path to whole slide images and annotations 
deploy_dataPath = './data/';
allFiles = dir([deploy_dataPath,'*.svs']);
datxml = './data/sedeen/';

layerNum = numel(myNet.Layers);
bsize = myNet.Layers(1).InputSize; 
bsize = bsize(1:2)-2*border; 
rmov = ceil(border(1)/bsize(1));  

% Process each whole slide image
for i = 1:numel(allFiles) 
    tic
     if exist(['.\output_masks\mask_out', '_', allFiles(i).name(1:end-4), '.mat'], 'file')
         continue;
     end
    currFilePath = [deploy_dataPath, allFiles(i).name];
    slidePtr = openslide_open(currFilePath);

    % Check if annotation exists for region selection
    if exist([datxml,  allFiles(i).name(1:end-4),'.session.xml'])
        xmlout = parseXML([datxml,  allFiles(i).name(1:end-4),'.session.xml']);
        [x1,x2,y1,y2] = read_xml(xmlout);
        judge=1;
        width=x2-x1;
        height=y2-y1;
        [mppX, mppY, width, height] = openslide_get_slide_properties(slidePtr);
    else
        judge=0;
        x1=0;y1=0;
        x2=0;y2=0;
        [mppX, mppY, width, height] = openslide_get_slide_properties(slidePtr);
    end
    imsetsize=1000;
    ind = floor(imsetsize/bsize(1));
    max_col = floor(double(width) / ((ind - 3) * bsize(1)));
    max_row = floor(double(height) / ((ind - 3) * bsize(1)));
    nPat = (max_col-1) * (max_row-1);  
    
    mask_total = cell(1, nPat);
    bar = waitbar(0, 'Loading data...');
    count1 = 0;

    % Process each tile in the grid
    for row = 1:max_row-1 
        for col = 1:max_col-1 
            
             % Update progress bar
            count1 = count1 + 1;
            str = ['Processing...', num2str(count1/nPat*100), '%'];
            waitbar(count1/nPat, bar, str);
            
            subheight = imsetsize;
            subwidth = imsetsize;
            left = rmov;
            right1 = ind-rmov;
            right2 = ind-rmov;
            
            %% Handle boundary cases
            if col == max_col - 1 && row ~= max_row - 1
                
                subwidth = width - ((ind - 3) * bsize(1)) * (max_col - 2);    
                right2 = floor(double(subwidth)/bsize(1))-rmov;
            end
            if row == max_row - 1 && col ~= max_col - 1
                subheight = height - ((ind - 3) * bsize(1)) * (max_row - 2);
                right1 = floor(double(subheight)/bsize(1))-rmov;
            end
            if row == max_row - 1 && col == max_col - 1
                subwidth = width - ((ind - 3) * bsize(1)) * (max_col - 2); 
                subheight = height - ((ind - 3) * bsize(1)) * (max_row - 2);
                right1 = floor(double(subheight)/bsize(1))-rmov;
                right2 = floor(double(subwidth)/bsize(1))-rmov;
            end
           %% Calculate tile coordinates
            x=x1+(col-1) * ((ind - 3) * bsize(1));
            y=y1+(row-1) * ((ind - 3) * bsize(1));
            if x<0 
                x=0;
            end
            if y<0 
                y=0;
            end
            if x+subwidth>width
                x=width-subwidth;
            end
            if y+subheight>height
                y=height-subheight;
            end
            % Read region from whole slide image
            result = openslide_read_region(slidePtr, x, y, subwidth, subheight, 0);
            % Check if center point is within ROI polygon
            if judge
                judge1 = judge_point(x+(imsetsize/2),y+(imsetsize/2),xmlout);
                if judge1
                result = result(: ,:, 2:4);
                else 
                result = result(: ,:, 2:4);
                result = zeros(size(result));
                end
            else
                result = result(: ,:, 2:4);
            end
            count2 = 0;
            imgset = [];
            for p = left:right1
                for q = left:right2
                    count2 = count2 + 1;
                    if (right1+1)*bsize(1)+border(1) > subheight
                       result(subheight+1:(right1+1)*bsize(1)+border(1), :, :) = 0;
                    end
                    if (right2+1)*bsize(1)+border(1) > subwidth
                       result(:, subwidth+1:(right2+1)*bsize(1)+border(1), :) = 0;
                    end
                    imgset(:, :, :, count2) = result(p*bsize(1)-border(1)+1:(p+1)*bsize(1)+border(1), q*bsize(1)-border(1)+1:(q+1)*bsize(1)+border(1), :); 
                    % Extract patch with overlap border ï¼?72+76*2=224ï¼?
                end
            end

            % Perform classification on patch batch
            [labels, score] = classify(myNet, imgset); 
            
            mid1 = zeros((right1-1)*(right2-1), 1, 8);
            mid3 = zeros(right1-1, right2-1, 8);
            for j = 1:8
                mid1(:, :, j) = score(:, j);
            end
            mid2 = reshape(mid1, right2-1, right1-1, 8);
            for k = 1:8
                mid3(:, :, k) = mid2(:, :, k)';
            end
            mask_total{count1} = mid3;
        end
    end
    % Reconstruct full slide mask
    mask = cell(max_row-1, max_col-1);
    count = 0;
    for m = 1:max_row-1
        for n = 1:max_col-1
            count = count + 1;
            mask{m, n} = mask_total{count};
        end
    end
       mask_out = cell2mat(mask);
    [rgbout, currstats] = mask8toRGB(mask_out, colors);
    
    allstats(i,:) = currstats(:);
    allnames{i} = allFiles(i).name;
    
    save (['./output_masks/mask_out', '_', allFiles(i).name(1:end-4)], 'mask_out');
    save (['./output_rgbout/rgbout', '_', allFiles(i).name(1:end-4)], 'rgbout');
  
    imwrite(rgbout, ['./output_images/mask', '_', allFiles(i).name(1:end-4), '_VGG.tif']);
    toc
end

statstable = [array2table(allnames', 'VariableNames' ,{'ID'}),...
    array2table(allstats, 'VariableNames', tissuenames)];

writetable(statstable,'ResultsTable.xlsx');

openslide_close(slidePtr)
clear slidePtr;
openslide_unload_library(); 
