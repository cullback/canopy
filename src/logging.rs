use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Initialize the global tracing subscriber with an indicatif layer.
///
/// Log output is routed through indicatif's `MultiProgress` so that log
/// lines appear above any active progress bars. Uses `try_init()` so that
/// repeated calls are safe no-ops.
pub fn init_logging() {
    let indicatif_layer = tracing_indicatif::IndicatifLayer::new();
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with(indicatif_layer)
        .try_init();
}
