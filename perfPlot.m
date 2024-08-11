clear
close all;
clc

addpath('./util');
addpath(('./rstEval'));
attPath = '.\UAV123_10fps\anno\att\'; % The folder that contains the annotation files for sequence attributes

%OTB100
%attName={'illumination variation'	'out-of-plane rotation'	'scale variation'	'occlusion'	'deformation'	'motion blur'	'fast motion'	'in-plane rotation'	'out of view'	'background clutter' 'low resolution'};
%attFigName={'illumination_variations'	'out-of-plane_rotation'	'scale_variations'	'occlusions'	'deformation'	'blur'	'abrupt_motion'	'in-plane_rotation'	'out-of-view'	'background_clutter' 'low_resolution'};

%UAV123_10fps
attName={'Scale Variation' 'Aspect Ratio Change' 'Low Resolution' 'Fast Motion' 'Full Occlusion' 'Partial Occlusion' 'Out-of-View' 'Background Clutter' 'Illumination Variation' 'Viewpoint Change' 'Camera Motion' 'Similar Object'};
attFigName={'SV'	'ARC'	'LR'	'FM'	'FOC'	'POC'	'OV'	'BC'	'IV'	'VC'	'CM'	'SOB'};

plotDrawStyleAll = {   
    struct('color',[1,0,0],'lineStyle', '-', 'lineWidth', 0.5),... % red
    struct('color',[0,1,0],'lineStyle', '-', 'lineWidth', 0.5),... % green
    struct('color',[0,0,1],'lineStyle', '-', 'lineWidth', 0.5),... % blue
    struct('color',[1,1,0],'lineStyle', '-', 'lineWidth', 0.5),... % yellow
    struct('color',[1,0,1],'lineStyle', '-', 'lineWidth', 0.5),... % magenta
    struct('color',[0,1,1],'lineStyle', '-', 'lineWidth', 0.5),... % cyan
    struct('color',[0.5,0.5,0.5],'lineStyle', '-', 'lineWidth', 0.5),... % gray-25%
    struct('color',[0,0,0],'lineStyle', '-', 'lineWidth', 0.5),... % black
    struct('color',[0,0.5,0.5],'lineStyle', '-', 'lineWidth', 0.5),... % teal
    struct('color',[0.5,0,0.5],'lineStyle', '-', 'lineWidth', 0.5),... % purple
    struct('color',[0.75,0,0.25],'lineStyle', '-', 'lineWidth', 0.5),... % maroon
    struct('color',[0.25,0.75,0.75],'lineStyle', '-', 'lineWidth', 0.5),... % turquoise
    struct('color',[0.5,0,0],'lineStyle', '-', 'lineWidth', 0.5),... % dark red
    struct('color',[0.5,0.5,0],'lineStyle', '-', 'lineWidth', 0.5),... % olive
    struct('color',[0,0.5,0],'lineStyle', '-', 'lineWidth', 0.5),... % dark green
    struct('color',[0,0.5,0.5],'lineStyle', '-', 'lineWidth', 0.5),... % dark turquoise
    struct('color',[0,0,0.5],'lineStyle', '-', 'lineWidth', 0.5),... % navy
    struct('color',[0.5,0,0.5],'lineStyle', '--', 'lineWidth', 0.5),... % light purple
    struct('color',[0.5,0.5,0.5],'lineStyle', '--', 'lineWidth', 0.5),... % gray-50%
    struct('color',[0.75,0.75,0.75],'lineStyle', '--', 'lineWidth', 0.5),... % gray-75%
    struct('color',[0.75,0,0],'lineStyle', '--', 'lineWidth', 0.5),... % light red
    struct('color',[0.75,0.75,0],'lineStyle', '--', 'lineWidth', 0.5),... % light olive
    struct('color',[0,0.75,0],'lineStyle', '--', 'lineWidth', 0.5),... % light green
    struct('color',[0,0.75,0.75],'lineStyle','--','lineWidth', 0.5),... % light turquoise
    struct('color',[0,0,0.75],'lineStyle','--','lineWidth', 0.5)};  % dark blue

plotDrawStyle10={   struct('color',[1,0,0],'lineStyle','-'),...
    struct('color',[0,1,0],'lineStyle','--'),...
    struct('color',[0,0,1],'lineStyle',':'),...
    struct('color',[0,0,0],'lineStyle','-'),...%    struct('color',[1,1,0],'lineStyle','-'),...%yellow
    struct('color',[1,0,1],'lineStyle','--'),...%pink
    struct('color',[0,1,1],'lineStyle',':'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','-'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','--'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle',':'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','-'),...%Turquoise
    };

seqs=configSeqs;

trackers=configTrackers;

% seqs = seqs(1:10);
% trackers = trackers(1:10);

numSeq=length(seqs);
numTrk=length(trackers);

nameTrkAll=cell(numTrk,1);
for idxTrk=1:numTrk
    t = trackers{idxTrk};
    nameTrkAll{idxTrk}=t.namePaper;
end

nameSeqAll=cell(numSeq,1);
numAllSeq=zeros(numSeq,1);

att=[];
for idxSeq=1:numSeq
    s = seqs{idxSeq};
    nameSeqAll{idxSeq}=s.name;
    
    s.len = s.endFrame - s.startFrame + 1;
    
    numAllSeq(idxSeq) = s.len;
    
    att(idxSeq,:)=load([attPath s.name '.txt']);
end

attNum = size(att,2);

figPath = '.\figs\final_2\';

perfMatPath = '.\perfMat\final_2\';

if ~exist(figPath,'dir')
    mkdir(figPath);
end

if ~exist(perfMatPath,'dir')
    mkdir(perfMatPath);
end

metricTypeSet = {'error', 'overlap'};
evalTypeSet = {'OPE'}; %'SRE', 'OPE'

rankingType = 'threshold'; %AUC, threshold

rankNum = 18;%number of plots to show

if rankNum == 10
    plotDrawStyle=plotDrawStyle10;
else
    plotDrawStyle=plotDrawStyleAll;
end

thresholdSetOverlap = 0:0.05:1;
thresholdSetError = 0:50;

for i=1:length(metricTypeSet)
    metricType = metricTypeSet{i};%error,overlap
    
    switch metricType
        case 'overlap'
            thresholdSet = thresholdSetOverlap;
            rankIdx = 11;
            xLabelName = 'Overlap threshold';
            yLabelName = 'Success rate';
        case 'error'
            thresholdSet = thresholdSetError;
            rankIdx = 21;
            xLabelName = 'Location error threshold';
            yLabelName = 'Precision';
    end  
        
    if strcmp(metricType,'error')&strcmp(rankingType,'AUC')
        continue;
    end
    
    tNum = length(thresholdSet);
    
    for j=1:length(evalTypeSet)
        
        evalType = evalTypeSet{j};%SRE, TRE, OPE
        
        plotType = [metricType '_' evalType];
        
        switch metricType
            case 'overlap'
                titleName = ['Success plots of ' evalType];
            case 'error'
                titleName = ['Precision plots of ' evalType];
        end

        dataName = [perfMatPath 'aveSuccessRatePlot_' num2str(numTrk) 'alg_'  plotType '.mat'];
        
        % If the performance Mat file, dataName, does not exist, it will call
        % genPerfMat to generate the file.
        if ~exist(dataName)
            genPerfMat(seqs, trackers, evalType, nameTrkAll, perfMatPath);
        end        
        
        load(dataName);
        numTrk = size(aveSuccessRatePlot,1);        
        
        if rankNum > numTrk | rankNum <0
            rankNum = numTrk;
        end
        
        figName= [figPath 'quality_plot_' plotType '_' rankingType];
        idxSeqSet = 1:length(seqs);
        
        % draw and save the overall performance plot
        figure;
        plotDrawSave(numTrk,plotDrawStyle,aveSuccessRatePlot,idxSeqSet,rankNum,rankingType,rankIdx,nameTrkAll,thresholdSet,titleName, xLabelName,yLabelName,figName,metricType);
        tightfig;
        savefig(gcf, [figName, '.fig']);

        % draw and save the performance plot for each attribute
        attTrld = 0;
        for attIdx=1:attNum
            
            idxSeqSet=find(att(:,attIdx)>attTrld);
            
            if length(idxSeqSet) < 2
                continue;
            end
            disp([attName{attIdx} ' ' num2str(length(idxSeqSet))])
            
            figName=[figPath attFigName{attIdx} '_'  plotType '_' rankingType];
            titleName = ['Plots of ' evalType ': ' attName{attIdx} ' (' num2str(length(idxSeqSet)) ')'];
            
            switch metricType
                case 'overlap'
                    titleName = ['Success plots of ' evalType ' - ' attName{attIdx} ' (' num2str(length(idxSeqSet)) ')'];
                case 'error'
                    titleName = ['Precision plots of ' evalType ' - ' attName{attIdx} ' (' num2str(length(idxSeqSet)) ')'];
            end
            
            figure;
            plotDrawSave(numTrk,plotDrawStyle,aveSuccessRatePlot,idxSeqSet,rankNum,rankingType,rankIdx,nameTrkAll,thresholdSet,titleName, xLabelName,yLabelName,figName,metricType);
            savefig(gcf, [figName, '.fig']);
            close all;
        end        
    end
end
