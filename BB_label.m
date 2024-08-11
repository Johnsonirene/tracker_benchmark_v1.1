% ������Ƶ��bounding box��label�������ʱƥ�䣬��߶��μӹ�Ч�ʣ�ע�����£�
% - ��Ҫ�� .\util\configTrackers ���ʵ��
% - ����ѡ��Ϊ saveFig

clear;
close all;
warning off all;

paperTitle = 'BB_label_fews'; % ��ԵĻ�����ڿ����ƺ͸�����
saveFig = true; % true=�����ͼ���, false=������

BBlabel_path = ['.\dataAnaly\', paperTitle, '\'];
if ~exist(BBlabel_path, 'dir')
    mkdir(BBlabel_path);
end
addpath('./util');

trks = configTrackers;
plotSetting;
LineWidth = 2;
plotDrawStyle = plotSetting;
location = length(trks); % �Լ���tracker����configTrackers�����󣬻�ͼʱ�Ż�λ������ͼ��
plotDrawStyle(:,[location,1]) = plotDrawStyle(:,[1,location]); % ���԰Ѻ�ɫ��ע�ŵ���location��

len=30;
x=-50;
y=0.05;
k=0;
ResutlBB_label = figure;

% for i = 1 : length(trks)
for i = length(trks) : -1 : 1
    hold on
    set(gca,'fontname', 'Times New Roman','FontSize',16);
    LineStyle = plotDrawStyle{i}.lineStyle;
    Color = plotDrawStyle{i}.color;
    tracker_name = trks{i}.namePaper;
    plot([x,x+len],[y,y],'linestyle',LineStyle,'color',Color,'linewidth',LineWidth);
    text(x+len+2,y-0.,tracker_name,'fontname', 'Times New Roman','FontSize',10,'Interpreter','tex');
    text_width = get(text, 'Extent');
    text_width = text_width(3); % Extent���Եĵ�����ֵ�ǿ��
    x=x+60+10*strlength(tracker_name);
    k=k+1;
    if mod(k,6)==0
        k=k-5;
        y=y-0.1;
        x=0;
    end
    if length(trks) == 1
       set(gca, 'XLim',[0 x+len]);
    end
end
axis off
tightfig;

if saveFig == true
    saveDir = [BBlabel_path, 'BBlabel.pdf'];
    print(gcf,'-dpdf',saveDir);
    fprintf('��������Ƶ�б߿�ı�ǩ��λ�� %s\n', saveDir);
end

rmpath('./util');