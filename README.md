# papers-from-scratch

Hard research papers, implemented from scratch. No framework shortcuts — build the actual mechanism (autograd, attention, tokenizer, optimizer, etc.), not a wrapper around an existing library. This list is not capped; more papers get added over time.

**Browse: [codersacademy006.github.io/papers-from-scratch](https://codersacademy006.github.io/papers-from-scratch/)** — clickable index into every journal, one paper at a time.

Implemented so far: 4

## A. Core autodiff & training infra
- [x] 01 [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (Rumelhart, Hinton, Williams 1986) — `01_backprop_autograd/` ([journal](01_backprop_autograd/JOURNAL.md))
- [x] 02 [Dropout](https://arxiv.org/abs/1207.0580) (Hinton et al 2012) — `02_dropout/` ([journal](02_dropout/JOURNAL.md))
- [x] 03 [Batch Normalization](https://arxiv.org/abs/1502.03167) (2015) — `03_batchnorm/` ([journal](03_batchnorm/JOURNAL.md))
- [x] 04 [Adam](https://arxiv.org/abs/1412.6980) (2014) — `04_adam/` ([journal](04_adam/JOURNAL.md))
- [ ] 05 [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016) — `05_layernorm/`

## B. CNN / vision classics
- [ ] 06 [LeNet-5](https://ieeexplore.ieee.org/document/726791) (LeCun et al 1998) — `06_lenet5/`
- [ ] 07 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) (2012) — `07_alexnet/`
- [ ] 08 [VGGNet](https://arxiv.org/abs/1409.1556) (2014) — `08_vggnet/`
- [ ] 09 [ResNet](https://arxiv.org/abs/1512.03385) (2015) — `09_resnet/`
- [ ] 10 [U-Net](https://arxiv.org/abs/1505.04597) (2015) — `10_unet/`
- [ ] 11 [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) (2020) — `11_vit/`

## C. Detection
- [ ] 12 [YOLO](https://arxiv.org/abs/1506.02640) (2015) — `12_yolo/`
- [ ] 13 [Faster R-CNN](https://arxiv.org/abs/1506.01497) (2015) — `13_faster_rcnn/`

## D. RNN / sequence models
- [ ] 14 [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber 1997) — `14_lstm/`
- [ ] 15 [GRU (RNN Encoder-Decoder)](https://arxiv.org/abs/1406.1078) (Cho et al 2014) — `15_gru/`
- [ ] 16 [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215) (2014) — `16_seq2seq/`
- [ ] 17 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau attention, 2014) — `17_bahdanau_attention/`

## E. Transformer / LLM core
- [ ] 18 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — `18_transformer/`
- [ ] 19 [BERT](https://arxiv.org/abs/1810.04805) (2018) — `19_bert/`
- [ ] 20 [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019) — `20_gpt2/`
- [ ] 21 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020) — `21_gpt3/`
- [ ] 22 [LoRA](https://arxiv.org/abs/2106.09685) (2021) — `22_lora/`
- [ ] 23 [Outrageously Large Neural Networks (Sparsely-Gated MoE)](https://arxiv.org/abs/1701.06538) (2017) — `23_moe_sparse/`

## F. Tokenization / embeddings
- [ ] 24 [Word2Vec](https://arxiv.org/abs/1301.3781) (2013) — `24_word2vec/`
- [ ] 25 [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) (2015) — `25_bpe_tokenizer/`

## G. Generative models
- [ ] 26 [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014) — `26_gan/`
- [ ] 27 [DCGAN](https://arxiv.org/abs/1511.06434) (2015) — `27_dcgan/`
- [ ] 28 [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) (2013) — `28_vae/`
- [ ] 29 [StyleGAN](https://arxiv.org/abs/1812.04948) (2018) — `29_stylegan/`
- [ ] 30 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020) — `30_ddpm/`
- [ ] 31 [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2020) — `31_ddim/`
- [ ] 32 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion, 2021) — `32_latent_diffusion/`
- [ ] 33 [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (2015) — `33_neural_style_transfer/`

## H. Reinforcement learning
- [ ] 34 [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602) (2013) — `34_dqn/`
- [ ] 35 [Asynchronous Methods for Deep RL (A3C)](https://arxiv.org/abs/1602.01783) (2016) — `35_a3c/`
- [ ] 36 [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (2017) — `36_ppo/`
- [ ] 37 [Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815) (2017) — `37_alphazero/`
- [ ] 38 [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015) — `38_double_dqn/`
- [ ] 39 [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (2015) — `39_gae/`

## I. Memory / graph / structured
- [ ] 40 [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (2014) — `40_neural_turing_machine/`
- [ ] 41 [Pointer Networks](https://arxiv.org/abs/1506.03134) (2015) — `41_pointer_networks/`
- [ ] 42 [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (2016) — `42_gcn/`
- [ ] 43 [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (2017) — `43_gat/`
- [ ] 44 [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (2018) — `44_neural_ode/`

## J. Multimodal / speech / representation learning
- [ ] 45 [CLIP](https://arxiv.org/abs/2103.00020) (2021) — `45_clip/`
- [ ] 46 [WaveNet](https://arxiv.org/abs/1609.03499) (2016) — `46_wavenet/`
- [ ] 47 [Tacotron 2](https://arxiv.org/abs/1712.05884) (2017) — `47_tacotron2/`
- [ ] 48 [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (2020) — `48_wav2vec2/`
- [ ] 49 [SimCLR](https://arxiv.org/abs/2002.05709) (2020) — `49_simclr/`
- [ ] 50 [Masked Autoencoders (MAE)](https://arxiv.org/abs/2111.06377) (2021) — `50_mae/`

## Rules
- Each folder holds one paper's from-scratch implementation + a short note on what broke/what the paper glossed over.
- No pulling in a library that already implements the thing you're building (e.g. no `torch.nn.MultiheadAttention` inside `18_transformer/`). Framework for tensors/autograd is fine; the mechanism itself is not.
- Check a box in this README when a folder's implementation runs and the note is written.
