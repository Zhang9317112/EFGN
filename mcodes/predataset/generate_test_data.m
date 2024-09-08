fileFolder=fullfile('F:\datasets\hyperspectralimage\dataset\Chikusei\test\');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
factor = 0.25;
img_size = 512;
bands = 128;
gt = zeros(numel(fileNames),img_size,img_size,bands);
ms = zeros(numel(fileNames),img_size*factor,img_size*factor,bands);
ms_bicubic = zeros(numel(fileNames),img_size,img_size,bands);
cd F:\datasets\hyperspectralimage\dataset\Chikusei\test\;
for i = 1:numel(fileNames)
    load(fileNames{i},'test');
    img_ms = single(imresize(test, factor));
    gt(i,:,:,:) = test;
    ms(i,:,:,:) = img_ms;
    ms_bicubic(i,:,:,:) = single(imresize(img_ms, 1/factor));
end

gt = single(gt);
ms = single(ms);
ms_bicubic = single(ms_bicubic);
save('F:\datasets\hyperspectralimage\chikusei\Chikusei_x4\tests\Chikusei_test.mat','gt','ms','ms_bicubic');