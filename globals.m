% Set up global variables used throughout the code
addpath learning;
addpath detection;
%addpath root;
addpath evaluation;
addpath visualization;
addpath helper;
addpath edge;
addpath geometry;
%addpath Bryan;

% directory for caching models, intermediate data, and results

global cachedir
if ~isempty(cachedir) && ~exist(cachedir,'dir')
    mkdir(cachedir);
end

%if ~exist([cachedir 'imflip/'],'dir'),
%  unix(['mkdir ' cachedir 'imflip/']);
%end
