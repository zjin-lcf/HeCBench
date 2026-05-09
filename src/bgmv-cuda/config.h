struct Config {
    // op selection
    char op[32];          // bgmv_shrink | bgmv_expand
    // dimensions
    int  num_tokens;      // batch_size * seq_length
    int  hidden_size;
    int  lora_rank;
    int  num_loras;
    // bench
    int  repeat;
    bool add_to_output;
    float scaling;
    bool vectorize;
};

static void usage(const char* argv0) {
    printf("Usage: %s [options]\n"
           "  --op       <bgmv_shrink|bgmv_expand\n"
           "  --tokens   <num_tokens>     (default 128)\n"
           "  --hidden   <hidden_size>    (default 4096)\n"
           "  --rank     <lora_rank>      (default 16)\n"
           "  --loras    <num_loras>      (default 4)\n"
           "  --repeat   <N>             (default 200)\n"
           "  --add_to_output            (add_to_output=true for expand ops)\n"
           "  --scaling  <float>         (default 1.0)\n"
           "  --vectorize                (select the vectorized kernels for shrink ops)\n",
           argv0);
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    strcpy(cfg.op, "bgmv_shrink");
    cfg.num_tokens  = 128;
    cfg.hidden_size = 4096;
    cfg.lora_rank   = 16;
    cfg.num_loras   = 4;
    cfg.repeat      = 200;
    cfg.add_to_output  = false;
    cfg.vectorize   = false;
    cfg.scaling     = 1.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--op")     && i+1<argc) strncpy(cfg.op, argv[++i], 31);
        else if (!strcmp(argv[i], "--tokens")  && i+1<argc) cfg.num_tokens  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden")  && i+1<argc) cfg.hidden_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rank")    && i+1<argc) cfg.lora_rank   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--loras")   && i+1<argc) cfg.num_loras   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat")  && i+1<argc) cfg.repeat      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--scaling") && i+1<argc) cfg.scaling     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--add_to_output"))    cfg.add_to_output = true;
        else if (!strcmp(argv[i], "--vectorize"))  cfg.vectorize = true;
        else { usage(argv[0]); exit(0); }
    }
    return cfg;
}

