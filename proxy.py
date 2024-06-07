def training_loop(epoch,model,dataloader,criterion, optimizer, miner = None, pre_miner = None):
    
    model.train()
    inf_batch_count = 0
    train_loss = 0
        
    if epoch == 1 or pre_miner is None:
        
        if pre_miner is not None:
            
            global informative_batches
            informative_batches = []
            proxy_labels = []
            proxies = []
            
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            images, labels, _ = batch

            num_places, num_images_per_place, C, H, W = images.shape

            images = images.view(num_places * num_images_per_place, C, H, W)
            labels = labels.view(num_places * num_images_per_place)

            descriptors = model(images.to(device)).cpu()

            if pre_miner is not None:
                with torch.no_grad(): # calcoliamo i proxies di ogni luogo
                    num_tensori, *_ = descriptors.shape
                    for i in range(0, num_tensori - 4 + 1, 4):
                        place_images = descriptors[i:i+4]
                        proxy = place_images.mean(dim=0).tolist()
                        proxies.append(proxy)
                        proxy_labels.append(int(labels[i]))            
            
            loss.backward() 
            optimizer.step()
            train_loss += loss.item()
            # print(f'Batch {batch_idx}, Loss: {loss.item()}')

        if pre_miner is not None:
            proxies = np.asarray(proxies, dtype = np.float32)
            proxy_labels = np.asarray(proxy_labels, dtype = np.int32)
            informative_batches = knn_search(proxies, proxy_labels) #restituisce una lista di liste contenente i place id di ogni informative batch
        
      
    else:
        proxy_labels = []
        proxies = []
        for batch in informative_batches:            
            optimizer.zero_grad()

            images = [dataset_train.__getitem__(label)[0] for label in batch]  # otteniamo le immagini corrispondenti ai place id nell'informative batch tramite la funzione getitem
            
            images = torch.stack(images)
            dimensions = images.shape
            labels = [torch.tensor(label).repeat(4) for label in batch]
            labels = torch.stack(labels)


            num_places, num_images_per_place, C, H, W = images.shape

            images = images.view(num_places * num_images_per_place, C, H, W)
            labels = labels.view(num_places * num_images_per_place)

            descriptors = model(images.to(device)).cpu()

            num_tensori, *_ = descriptors.shape
            with torch.no_grad(): # calcoliamo i proxies di ogni luogo
                for i in range(0, num_tensori - 4 + 1, 4):
                    place_images = descriptors[i:i+4]
                    proxy = place_images.mean(dim=0).tolist()
                    proxies.append(proxy)
                    proxy_labels.append(int(labels[i]))

            loss.backward() 
            optimizer.step()
            train_loss += loss.item()
            inf_batch_count += 1
            # print(f'Batch {batch_idx}, Loss: {loss.item()}')


        proxies = np.asarray(proxies, dtype = np.float32)
        proxy_labels = np.asarray(proxy_labels, dtype = np.int32)

        informative_batches = knn_search(proxies, proxy_labels) #restituisce una lista di liste contenente i place id di ogni informative batch

    train_loss = train_loss / len(dataloader)
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f}')
    #print(informative_batches, len(informative_batches),len(informative_batches[5]))
    # return train_loss