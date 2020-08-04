function [ Track ] = track_labels(XTag,Xk,model)
%track_labels.m is a function that finds estimated states with the same
%label and collects them into tracks.

%Pina Gruden, ISVR, 2017

IDlist= unique([XTag{:}]);
N=numel(IDlist); %number of all tracks
if N==0 %there are no tracks
    Track=[];
    return
end
%preallocate
Track(N).freq =[];
Track(N).chirp =[];
Track(N).time =[];
Track(N).label =[];
Track(N).ti =[];

t=0:model.dt:size(XTag,2)*model.dt-model.dt;

for k=1:size(XTag,2)
    
    if ~isempty(XTag{k})
        [ni, indxs] = histc(XTag{k}, IDlist); %gives indices of which IDs the tags belong to
        multiple = find(ni > 1); %gives indication wether there are estimates with the same ID
        
        %identify which IDs are repeated and which not
        ID=ismember(indxs, multiple); %Logical array
        
        %~~~~~~~~~~~~~~ For IDS that are NOT repeated  ~~~~~~~~~~~~~~~~~~
        indx=indxs(ID==0);
        for m=1:length(indx)
            Track(indx(m)).freq=[Track(indx(m)).freq;Xk{k}(1,indxs==indx(m))];
            Track(indx(m)).chirp=[Track(indx(m)).chirp;Xk{k}(2,indxs==indx(m))];
            Track(indx(m)).time=[Track(indx(m)).time; t(k)];
            Track(indx(m)).label=[Track(indx(m)).label;IDlist(indx(m))];
            Track(indx(m)).ti = [Track(indx(m)).ti,k]; %time step index
        end
        
        %~~~~~~~~~~~~~~ For IDS that ARE repeated  ~~~~~~~~~~~~~~~~~~
        if ~isempty(multiple)
            for m=1:length(multiple)
                %------------ Resolve based on distance ------------------
                
                detections = Xk{k}(:,indxs==multiple(m));
                %predictions:
                M=multiple(m);
                
                if isempty(Track(M).ti)
                    %if this is the first time this track/ID is detected, then no
                    %info is there,so just take a mean between the estimates to
                    %be the real estimate
                    meanest=mean(detections,2);
                    Track(M).freq=[Track(M).freq;meanest(1)];
                    Track(M).chirp=[Track(M).chirp;meanest(2)];
                    Track(M).time=[Track(M).time; t(k)];
                    Track(M).label=[Track(M).label;IDlist(M)];
                    Track(M).ti = [Track(M).ti,k];
                else %the track alredy exists
                    %compute chirp : using chirp
                    if  numel(Track(M).ti) <= 10
                        alpha =median(Track(M).chirp);
                    else
                        alpha = median(Track(M).chirp(end-10:end));
                    end
                    
                    %deterime how many time steps ago it was last detected
                    Nstp=k-Track(M).ti(end);
                    A=model.F; A(1,2)=A(1,2)*Nstp;
                    prediction=A*[Track(M).freq(end);alpha];
                    
                    dif=detections-prediction;
                    %compute Mahalanobis distance
                    sig = [100,0;0,1000]; %give covariance for freq and chirp
                    mah = sqrt(sum(dif'.^2/sig,2));
                    [~,i]=min(mah); %index of the component with min distance from expected location
                    
                    Track(M).freq=[Track(M).freq;detections(1,i)];
                    Track(M).chirp=[Track(M).chirp;detections(2,i)];
                    Track(M).time=[Track(M).time; t(k)];
                    Track(M).label=[Track(M).label;IDlist(M)];
                    Track(M).ti = [Track(M).ti,k]; %time step index
                end
            end
        end
    end
end
end



