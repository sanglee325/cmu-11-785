from calendar import EPOCH
from config import *

def calculate_levenshtein(h, y, lh, ly):

    h = h.permute(1,0,2)
    h_argmax = torch.argmax(h)
    batch_size = y.shape[0] # TODO

    for i in range(batch_size):  # Loop through each element in the batch

        h_sliced =  h_argmax[i]

        lend = (h_sliced == 29).nonzero(as_tuple=False)

        if lend.size(0)!=0:

            h_sliced = h_sliced[:lend[0]]
        else:
            h_sliced = h_sliced

        h_string = ''.join(index2letter[j.item()] for j in h_sliced[:]) # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string

        y_sliced =  y[i, :ly[i]][1:-1]# TODO: Do the same for y - slice off the padding with ly
        y_string =   ''.join(index2letter[j.item()] for j in y_sliced)# TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        dis = Levenshtein.distance(h_string, y_string)

        dist +=dis

    dist /= batch_size

    return dist


def train(epoch, model, train_loader, optimizer, criterion, scaler, scheduler):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    total_loss = 0

    for i, (x, y, lx, ly) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)
        outputs, length = model(x, lx)

        loss = criterion(outputs, y, length, ly)
        total_loss += float(loss)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        lr_rate = float(optimizer.param_groups[0]['lr'])

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(loss),
            lr="{:.04f}".format(lr_rate)
        )
        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar
    
    train_loss = total_loss / len(train_loader)    
    print("Epoch {}/{}: train loss {:.04f}, learning rate {:.04f}".format(
                                        epoch + 1, EPOCHS, train_loss, lr_rate))
        
    return train_loss, lr_rate

def validate(epoch, model, val_loader, criterion):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Validation')

    total_loss = 0
    total_dist = 0

    for i, (x, y, lx, ly) in enumerate(val_loader):
        x = x.to(device)
        
        with torch.no_grad():
            outputs, length = model(x, lx)
        dist = calculate_levenshtein(outputs,y,lx,ly)
        total_dist += dist

        loss = criterion(outputs, y, length, ly)
        total_loss += loss

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(loss),
            dist="{:.04f}".format(dist)
        )

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar

    val_loss = float(total_loss / len(val_loader))
    val_dist = float(total_dist / len(val_loader))
        
    print("Epoch {}/{}: validation loss {:.04f}, distance {:.04f}".format(epoch + 1, EPOCHS, val_loss, val_dist))

    return val_loss, val_dist

if __name__ == '__main__':
    logpath = ARGS.log_path
    logfile_base = f"{ARGS.name}_S{SEED}_B{BATCH_SIZE}_LR{LR}_E{EPOCHS}"
    logdir = logpath + logfile_base

    set_logpath(logpath, logfile_base)
    print('save path: ', logdir)
    
    train_loader, val_loader, test_loader = load_dataset(BATCH_SIZE)

    model = Seq2Seq(ARGS.input_dim, len(LETTER_LIST))

    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * EPOCHS))
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in EPOCHS:
        train(epoch, model, train_loader, optimizer, criterion, scheduler)
        validate(model, val_loader, criterion)