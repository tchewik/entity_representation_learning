local dataset_name = std.extVar("dataset_name");
local cuda_device = std.parseInt(std.extVar("cuda_device"));
local foldnum = std.extVar("foldnum");
local embedding_dim = 768;
local max_length = 25;
local dropout = 0.4;
local pretrained_model = std.extVar("pretrained_model");
local lr = std.parseJson(std.extVar("lr"));
local batch_size = std.parseInt(std.extVar("batch_size"));
local num_labels1 = std.parseInt(std.extVar("num_labels1"));
local num_labels2 = std.parseInt(std.extVar("num_labels2"));
local num_labels3 = std.parseInt(std.extVar("num_labels3"));
local num_labels4 = std.parseInt(std.extVar("num_labels4"));
local num_labels5 = std.parseInt(std.extVar("num_labels5"));


{
  vocabulary: {
    non_padded_namespaces: ["tokens", "labels1", "labels2", "labels3", "labels4", "labels5"]
  },
  dataset_reader:{
      type:  "src.custom_model.five_outputs_rdr.FiveOutputsTextClassificationJsonReader",
      num_labels1: num_labels1,
      num_labels2: num_labels2,
      num_labels3: num_labels3,
      num_labels4: num_labels4,
      num_labels5: num_labels5,
      tokenizer: {
        type: "pretrained_transformer",
        model_name: pretrained_model,
        max_length: max_length,
      },
      token_indexers: {
          tokens: {
            type: "pretrained_transformer",
            model_name: $.dataset_reader.tokenizer.model_name,
            max_length: max_length,
            namespace: 'tokens',
       },
      },
  },
  train_data_path: 'data/folds/' + foldnum + '/' + dataset_name + '_train_c_deberta_tokens.json',
  validation_data_path: 'data/folds/' + foldnum + '/' + dataset_name + '_val_c_deberta_tokens.json',
  //test_data_path: 'data/folds/' + foldnum + '/' + dataset_name + '_test_c_deberta_tokens.json',
  model: {
    type: 'src.custom_model.five_outputs_clf.FiveOutputsTextClassifier',
    with_connected_outputs: true,
    with_hierarchical_loss: true,
    data_file: 'data/amz_metadata/' + dataset_name + '.pkl',
    foldnum: std.parseInt(foldnum),
    dropout: dropout,
    num_labels1: num_labels1,
    num_labels2: num_labels2,
    num_labels3: num_labels3,
    num_labels4: num_labels4,
    num_labels5: num_labels5,
    text_field_embedder: {
        token_embedders: {
            tokens: {
              type: "pretrained_transformer",
              model_name: $.dataset_reader.tokenizer.model_name,
              max_length: max_length,
            },
        },
    },
    seq2vec_encoder: {
       type: "cls_pooler",
       embedding_dim: embedding_dim,
    },
    device: $.trainer.cuda_device,
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      sorting_keys: ["tokens"],
      batch_size: batch_size,
    },
  },
  trainer: {
    callbacks: [
        {
            type: "wandb",
            project: "prodmatching-ckp",
            name: dataset_name + "_" + foldnum,
        }
    ],
    type: "gradient_descent",
    use_amp: true,
    optimizer: {
      type: "huggingface_adamw",
      lr: lr,
      weight_decay: 0.1,
    },
    learning_rate_scheduler: {
      type: "slanted_triangular",
      cut_frac: 0.06
    },
    validation_metric: '+f1_avg',
    num_serialized_models_to_keep: 1,
    num_epochs: 30,
    patience: 3,
    cuda_device: cuda_device,
  },
}
