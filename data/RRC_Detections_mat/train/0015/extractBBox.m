for i = 1:376
    i
    recs = detections{i};
    maxIdx = find(recs(:,5)>0.4);    
    goodRecs = recs(maxIdx,:);

   dlmwrite(sprintf('../../../RRC_Detections_mat/train/0015/%06d.txt',i-1), goodRecs,'delimiter', ' ');
    
%     imshow(imread(sprintf('/media/junaid/21521fed-7c6d-46bf-a767-bdd1c8df1744/junaid/Research@IIITH/ICRA_18_MOT/data/KITTI_TRAIN_SEQ/0015/image_02/%06d.png',i-1)));
%     hold on;
%     for j = 1:size(goodRecs,1)
%         r = goodRecs(j,:);
%         rectangle('Position',[r(1), r(2), r(3)-r(1), r(4)-r(2)],'EdgeColor','y','LineWidth',3)        
%     end
    
    
%     hold off;
%     pause(0.3);
end