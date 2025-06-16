clc
clear
close all

openslide_load_library();

addpath(genpath('./subroutines/'));
mkdir('./output_images');
mkdir('./output_masks');
mkdir('./output_rgbout');
load('.\trained_vgg19_model\new_zhongshan_vgg19_model_1.mat');

colors = [ 192, 192, 192; 0, 255, 255; 255, 165, 0; 0, 0, 255; 255, 0, 255; 255, 255, 0; ...
    0, 255, 0; 255, 0, 0]/255;%灰青橙蓝紫黄绿红

tissuenames = {'ADI','BACK','INT','LYM','MUS','NORM','STR','TUM'};
border = [76 76]; % border是 patch 重叠的边缘

deploy_dataPath = 'E:\HWH\PAAD_proj\PAADNEW_zhongshan\new_zhongshan_87\';
allFiles = dir([deploy_dataPath,'*.svs']);

layerNum = numel(myNet.Layers);
bsize = myNet.Layers(1).InputSize; 
bsize = bsize(1:2)-2*border; % bsize是不重叠读取时候的patch边长，为72
rmov = ceil(border(1)/bsize(1));   %ceil（）向上取整

for i = 1:numel(allFiles) 
    tic
    if exist(['E:\HWH\PAAD_proj\PAAD_TCGA\output_masks\mask_out', '_', allFiles(i).name(1:end-4), '.mat'], 'file')
        continue;
    end
    currFilePath = [deploy_dataPath, allFiles(i).name];
    slidePtr = openslide_open(currFilePath);
    [mppX, mppY, width, height] = openslide_get_slide_properties_1(slidePtr);

    ind = floor(2000/bsize(1));
   
    max_col = floor(double(width) / ((ind - 3) * bsize(1)));
    max_row = floor(double(height) / ((ind - 3) * bsize(1)));
    nPat = (max_col-1) * (max_row-1);  %一共有 nPat 个 2000×2000 的patch
    
    mask_total = cell(1, nPat);
    bar = waitbar(0, '读取数据中');
    count1 = 0;
  
    for row = 1:max_row-1 
        for col = 1:max_col-1 
            
            %读条用
            count1 = count1 + 1;
            str = ['计算中...', num2str(count1/nPat*100), '%'];
            waitbar(count1/nPat, bar, str);
            
            subheight = 2000;
            subwidth = 2000;
            left = rmov;
            right1 = ind-rmov;
            right2 = ind-rmov;
            
           %% 防止超过边界
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
           %%
            result = openslide_read_region_1(slidePtr, (col-1) * ((ind - 3) * bsize(1)), (row-1) * ((ind - 3) * bsize(1)),...
                subwidth, subheight, 0);
            
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
                    % bsize加上border重叠的边缘的两倍就是提取出来的patch的大小（72+76*2=224）
                end
            end

            [labels, score] = classify(myNet, imgset); %2000×2000的patch（分成 count2 个有重叠的 224×224小patch）作为一个 imgset
            
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

writetable(statstable,'new_zhongshan_87_ResultsTable.xlsx');

openslide_close(slidePtr)
clear slidePtr;
openslide_unload_library(); 
