% Script that produces side by side images of mean cat and mean dog
% in axis and saves the image to child folder.

% Author: Jernej Vivod

% Get images in folder and build set of algorithms names.
files = dir("*.png");
algs = cell(1, length(files));

count = 1;
for file = files'
    file_name_split = split(file.name, '_');
    alg_name = file_name_split{1};
    algs{count} = alg_name;
    count = count + 1;
end
algs = unique(algs);    


% Go over algorithm names and concatenate images, position
% figure and save to child folder.
for idx = 1:length(algs)
    I_cat = imread(algs{idx} + "_cat.png");
    I_dog = imread(algs{idx} + "_dog.png");
    fig = figure();
    subplot(1, 2, 1);
    imagesc(I_cat);
    subplot(1, 2, 2);
    imagesc(I_dog);
    fig.set('Position', [1 541 960 450]);
    sgtitle("       " + algs{idx});
    saveas(fig, "./processed/" + algs{idx} + ".png");
end
    
