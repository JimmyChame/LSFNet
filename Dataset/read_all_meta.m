close all
clear all
clc

path = '/DataSet/fivek_dataset/RGGB/train/';
fileFolder = fullfile(path);
dirOutput=dir(fullfile(fileFolder,'*.dng'));
fileNames = {dirOutput.name}';


for i = 1%size(fileNames, 1)
    image_name = fileNames{i};
    metadata = imfinfo(char([path, image_name]));

    % Black and White
    if isfield(metadata,'BlackLevel')
        meta.black = metadata.BlackLevel;
        meta.saturation = metadata.WhiteLevel;
        if isfield(metadata, 'BlackLevelDeltaV')
            bldv = metadata.BlackLevelDeltaV;
            meta.black = meta.black + round(mean(bldv(:)));
        end
        if isfield(metadata, 'BlackLevelDeltaH')
            bldh = metadata.BlackLevelDeltaH;
            meta.black = meta.black + round(mean(bldh(:)));
        end
    else
        meta.black = metadata.SubIFDs{1,1}.BlackLevel;
        meta.saturation = metadata.SubIFDs{1,1}.WhiteLevel;
        try
            bldv = metadata.SubIFDs{1,1}.BlackLevelDeltaV;
            meta.black = meta.black + round(mean(bldv(:)));
        catch
        end
        try
            bldh = metadata.SubIFDs{1,1}.BlackLevelDeltaH;
            meta.black = meta.black + round(mean(bldh(:)));
        catch
        end
    end
    %{
    % CFA patter
    cfachar = ['R', 'G', 'B'];
    cfaidx = [];
    if isfield(metadata,'UnknownTags')
        ut = metadata.UnknownTags;
        if size(ut, 1) >= 2
            cfaidx = ut(2).Value;
        end
    elseif isfield(metadata.extra, 'CFAPattern2')
        cfap = metadata.extra.CFAPattern2;
        cfacells = strsplit(cfap, ' ');
        cfaidx = str2num(char(cfacells))';
    else
        error('Could not find CFA Pattern');
    end
    if length(cfaidx) ~= 4
        cfaidx = metadata.SubIFDs{1, 1}.UnknownTags(2).Value;
    end
    cfaidx = uint8(cfaidx);
    meta.cfapattern = cfachar(cfaidx + 1);
    [ ~, meta.cfapattern ] = cfa_pattern(metadata);
    %}
    meta.cfapattern = 'RGGB';
    % white balance
    if isfield(metadata, 'AsShotNeutral')
        meta.wb = metadata.AsShotNeutral;
    else
        continue;
    end
    % xyz2cam
    meta.xyz2cam = metadata.ColorMatrix2;
    % orientation
    if isfield(metadata, 'Orientation')
        meta.orientation = metadata.Orientation;
    end
    
    save([path, image_name(1:end-4), '.mat'], 'meta')
    fprintf(num2str(i))      
end
