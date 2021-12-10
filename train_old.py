import pickle
from tqdm import tqdm

from VSRN import VSRN
from metrics_utils import validate
from dao import save_checkpoint, get_config
from data import get_dataloader
from train_utils import adjust_learning_rate
from Vocabulary import Vocabulary


def main():
    # Load Vocabulary Wrapper  # todo make new vocab
    vocab = pickle.load(open('checkpoints_and_vocabs/f30k_precomp_vocab.pkl', 'rb'))

    train_loader = get_dataloader(
        type='train',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_train_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_train_emb_CLIP_fixed.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_train_emb_CLIP_fixed.npy',
        images_path='',
        vocab=vocab
    )
    val_loader = get_dataloader(
        type='eval',
        annotations_map_name='checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json',
        image_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fixed.npy',
        ocr_embeddings_name='precomputed_embeddings/final_all_test_emb_CLIP_fixed.npy',
        images_path='',
        vocab=vocab
    )

    params = get_config('inference_config.yaml')
    params['vocab_size'] = len(vocab)
    model = VSRN(params['grad_clip'],
                 params['image_embedding_dim'],
                 params['gcn_embedding_size'],
                 params['vocab_size'],
                 params['caption_encoder_word_dim'],
                 params['caption_encoder_num_layers'],
                 params['caption_encoder_embedding_size'],
                 params['dim_vid'],  # todo вероятно это то же самое, что и gcn_embedding_size, но надо проверить
                 params['dim_caption_generation_hidden'],
                 params['input_dropout_p_caption_generation_enc'],
                 params['rnn_type_caption_generation_enc'],
                 params['rnn_dropout_p_caption_generation_enc'],
                 params['bidirectional_enc'],
                 params['max_caption_len'],
                 params['dim_word_caption_generation'],
                 params['input_dropout_p_caption_generation_dec'],
                 params['rnn_type_caption_generation_dec'],
                 params['rnn_dropout_p_caption_generation_dec'],
                 params['bidirectional_dec'],
                 params['margin'],
                 params['measure'],
                 params['max_violation'],
                 params['learning_rate'])

    # Train the Model
    best_rsum = 0

    for epoch in range(params['num_epochs']):
        adjust_learning_rate(params['learning_rate'],
                             params['lr_update'],
                             model.optimizer,
                             epoch)

        # train for one epoch
        best_rsum = train(train_loader, model, epoch, val_loader, params['log_step'], params['measure'], best_rsum)
        print(best_rsum)
        # raise ValueError
        # evaluate on validation set
        rsum = validate(params['log_step'],
                        params['measure'],
                        val_loader,
                        model)  # todo подумать как реализовать разделение на вал сет. Пока не трогаю
        print(best_rsum)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum  # todo Что-то тут странно, чувствую, есть косяк
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            # 'opt': opt,
            # 'Eiters': model.Eiters,
        },
            is_best,
            prefix='runs/log/')


def train(train_loader, model, epoch, val_loader, log_step, measure, best_rsum):  # todo add time tracking
    # switch to train mode
    model.train_start()
    for i, train_data in tqdm(enumerate(train_loader)):
        model.train_start()

        # measure data loading time
        # data_time.update(time.time() - end)

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        #
        # Print log info
        # if model.Eiters % opt.log_step == 0:
        #     logging.info(
        #         'Epoch: [{0}][{1}/{2}]\t'
        #         '{e_log}\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #         .format(
        #             epoch, i, len(train_loader), batch_time=batch_time,
        #             data_time=data_time, e_log=str(model.logger)))
        #     print(
        #         'Epoch: [{0}][{1}/{2}]\t'
        #         '{e_log}\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #         .format(
        #             epoch, i, len(train_loader), batch_time=batch_time,
        #             data_time=data_time, e_log=str(model.logger)))

        # # validate at every val_step
        if i % 100 == 0 and i > 0:
            # validate(opt, val_loader, model)

            # evaluate on validation set
            rsum = validate(log_step,
                            measure,
                            val_loader,
                            model)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                # 'opt': opt,
                # 'Eiters': model.Eiters,
            },
                is_best,
                prefix='runs/log/')

    return best_rsum


if __name__ == '__main__':
    main()
