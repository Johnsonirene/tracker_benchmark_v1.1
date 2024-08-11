
% 生成各跟踪器的跟踪图片序列，注意如下：
% 同 .\util\configSeqs.m 和 configTrackers.m 进行配合
% by lfl

close all;
clear, clc;
warning off all;

addpath('.\util\');

evalType = 'OPE'; % 'SRE', 'OPE'
resPath = ['.\results\\results_uav\results_' evalType '\']; % The folder containing the tracking results、
% ground_truth = dlmread('.\anno\bird1_3.txt');

% 可选项
paperTitle = 'Box'; % 针对的会议或期刊名称和作者
ownTrakerLineWidth = 1; % 自己的跟踪器线粗点
otherTrakcerLineWidth = 1;
visibleDraw = 'on'; % 'on':可视化绘图过程; 'off':不可视化，但能保存图片

drawPath = ['D:\ML\MA\experiment\', paperTitle, '\UAV123_10fps\videoImgs_uav3\']; % The folder that will stores the images with overlaid bounding box
if ~exist(drawPath, 'dir')
    mkdir(drawPath);
end

rstIdx = 1;

seqs = configSeqs;
trks = configTrackers;

if isempty(rstIdx)
    rstIdx = 1;
end

plotDrawStyle = plotSetting;
location = length(trks); % 自己的tracker放在configTrackers里的最后，画图时才会位于最上图层
plotDrawStyle(:,[location,1]) = plotDrawStyle(:,[1,location]); % 所以把红色标注放到第location个

lenTotalSeq = 0;
resultsAll=[];
trackerNames=[];
num_seqs = length(seqs);
fprintf('Total number of seqs: %d\n', num_seqs);
t_frames_make_st = clock;
for index_seq=1:length(seqs)
    seq = seqs{index_seq};
    seq_name = seq.name;
    
    seq_length = seq.endFrame-seq.startFrame+1; % size(rect_anno,1);
    lenTotalSeq = lenTotalSeq + seq_length;

    if seq_length <700
        for index_algrm=1:length(trks)
            algrm = trks{index_algrm};
            name = algrm.name;
            trackerNames{index_algrm}=name;
                   
            fileName = fullfile(resPath, [seq_name '_' name '.mat']);
        
            data = load(fileName);
            
            res = data.results{rstIdx};
            
            if ~isfield(res,'type')&&isfield(res,'transformType')
                res.type = res.transformType;
                res.res = res.res';
            end
                
            if strcmp(res.type,'rect')
                for i = 2:res.len
                    r = res.res(i,:);
                   
                    if (isnan(r) | r(3)<=0 | r(4)<=0)
                        res.res(i,:)=res.res(i-1,:);
                        %             results.res(i,:) = [1,1,1,1];
                    end
                end
            end
    
            resultsAll{index_algrm} = res;
    
        end
            
        nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
        
    %     pathSave = [drawPath seq_name '_' num2str(rstIdx) '/'];
        pathSave = [drawPath seq_name '\'];
        if ~exist(pathSave,'dir')
            mkdir(pathSave);
        end
        fprintf('Start making frames %d: %s', index_seq, seq_name);
        
        figure('Name','Drawing result BB','NumberTitle','off','Color','white','Visible',visibleDraw);
        for i = 1:seq_length
            image_no = seq.startFrame + (i-1);
            id = sprintf(nz,image_no);
            fileName = strcat(seq.path,id,'.',seq.ext);
            
            img = imread(fileName);
            
            imshow(img);
    
%             text(7, 37, ['#' id], 'Color', 'blue', 'FontSize', 24, 'BackgroundColor', 'w','EdgeColor', 'k');

            % draw groundtruth
            % gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
%             b_gt = ground_truth(i,:);
%             rectangle('Position', b_gt, 'EdgeColor', plotDrawStyle{26}.color, 'LineWidth', 1,'LineStyle', plotDrawStyle{26}.lineStyle);

            for j = 1:length(trks)
                LineStyle = plotDrawStyle{j}.lineStyle;
                
                if j == location % 自己的tracker线粗一点
                    LineWidth = ownTrakerLineWidth;
                else
                    LineWidth = otherTrakcerLineWidth;
                end
                
                switch resultsAll{j}.type
                    case 'rect'
                        rectangle('Position', resultsAll{j}.res(i,:), 'EdgeColor', plotDrawStyle{j}.color, 'LineWidth', LineWidth,'LineStyle',LineStyle);
                    case 'ivtAff'
                        drawbox(resultsAll{j}.tmplsize, resultsAll{j}.res(i,:), 'Color', plotDrawStyle{j}.color, 'LineWidth', LineWidth,'LineStyle',LineStyle);
                    case 'L1Aff'
                        drawAffine(resultsAll{j}.res(i,:), resultsAll{j}.tmplsize, plotDrawStyle{j}.color, LineWidth, LineStyle);                    
                    case 'LK_Aff'
                        [corner c] = getLKcorner(resultsAll{j}.res(2*i-1:2*i,:), resultsAll{j}.tmplsize);
                        hold on,
                        plot([corner(1,:) corner(1,1)], [corner(2,:) corner(2,1)], 'Color', plotDrawStyle{j}.color,'LineWidth',LineWidth,'LineStyle',LineStyle);
                    case '4corner'
                        corner = resultsAll{j}.res(2*i-1:2*i,:);
                        hold on,
                        plot([corner(1,:) corner(1,1)], [corner(2,:) corner(2,1)], 'Color', plotDrawStyle{j}.color,'LineWidth',LineWidth,'LineStyle',LineStyle);
                    case 'SIMILARITY'
                        warp_p = parameters_to_projective_matrix(resultsAll{j}.type,resultsAll{j}.res(i,:));
                        [corner c] = getLKcorner(warp_p, resultsAll{j}.tmplsize);
                        hold on,
                        plot([corner(1,:) corner(1,1)], [corner(2,:) corner(2,1)], 'Color', plotDrawStyle{j}.color,'LineWidth',LineWidth,'LineStyle',LineStyle);
                    otherwise
                        disp('The type of output is not supported!')
                        continue;
                end
            end        
            imwrite(frame2im(getframe(gcf)), [pathSave  num2str(i) '.png']);
        end
        clf
        fprintf(' End!\n');
        close all;
    end
end
t_frames_make_end = clock;
t_frames_make = etime(t_frames_make_end, t_frames_make_st);
fprintf('End making videos, elapsed time: %.2fs. All imgs are in: %s\n', t_frames_make, drawPath);

rmpath('.\util\');