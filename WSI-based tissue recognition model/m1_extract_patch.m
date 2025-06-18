
clear
close all
tic

openslide_load_library();

mkdir('./output_image');
datxml = dir('./images/*.xml');
datsvs = dir('./images/*.svs');
file = dir('./output_image2');

for i = 1:length(datxml)
    xmlout = parseXML(['./images/', datxml(i).name]);
    slidePtr = openslide_open(['./images/', datsvs(i).name]);
    for j = 2:2:length(xmlout.Children)
        mid1 = xmlout.Children(j).Children(4).Children;
        ind = 0;
        for k = 4:2:length(mid1)
            mid2 = mid1(k).Children(4).Children;
            ind = ind + 1;
            x_total = [];
            y_total = [];
            ind1 = 0; 
            for m = 2:2:length(mid2)
                ind1 = ind1 + 1;
                x_total(ind1, 1) = floor(str2double(mid2(m).Attributes(1).Value));
                y_total(ind1, 1) = floor(str2double(mid2(m).Attributes(2).Value));
            end
            x1 = min(x_total) - 50;
            x2 = max(x_total) + 50;
            y1 = min(y_total) - 50;
            y2 = max(y_total) + 50;
            xi = [x1, x2, x1, x2];
            yi = [y1, y1, y2, y2];
            subim = openslide_read_region(slidePtr, x1, y1, x2 - x1, y2 - y1, 0); 
            img = subim(:, :, 2:4);
            bw = roipoly(xi, yi, img, x_total, y_total);
            se1 = strel('square', 50);
            mask = imerode(bw, se1);
            patchsize = [224 224];
            num1 = floor(size(bw, 1) / patchsize(1));
            num2 = floor(size(bw, 2) / patchsize(2));
            ind2 = 0;

            for p = 1:num2
                for q = 1:num1
                    patch = bw((q-1)*224+1:q*224, (p-1)*224+1:p*224);
                    if patch(patchsize(1)/2:patchsize(1)/2+1, patchsize(2)/2:patchsize(2)/2+1) == 1
                        ind2 = ind2 + 1;
                        out_patch = img((q-1)*224+1:q*224, (p-1)*224+1:p*224, :);
                        if j/2 == 1
                            filename = file(9).name;
                        end
                        if j/2 == 2
                            filename = file(8).name;
                        end
                        if j/2 == 3
                            filename = file(5).name;
                        end
                        if j/2 == 4
                            filename = file(3).name;
                        end
                        if j/2 == 5
                            filename = file(7).name;
                        end
                        if j/2 == 6
                            filename = file(4).name;
                        end
                        if j/2 == 7
                            filename = file(6).name;
                        end
                        
                        imwrite(out_patch, ['./output_image/', filename, '/', datsvs(i).name(1:end-4),... 
                            '_', num2str(ind), '_', num2str(ind2), '.tif']);
                    end
                end
            end
        end 
    end
    openslide_close(slidePtr)
    clear slidePtr
end

openslide_unload_library(); 
toc