export var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["DEBUG"] = 0] = "DEBUG";
    LogLevel[LogLevel["INFO"] = 1] = "INFO";
    LogLevel[LogLevel["WARN"] = 2] = "WARN";
    LogLevel[LogLevel["ERROR"] = 3] = "ERROR";
    LogLevel[LogLevel["SILENT"] = 4] = "SILENT";
})(LogLevel || (LogLevel = {}));
export class Logger {
    constructor(config = {}) {
        this.config = {
            level: config.level ?? LogLevel.INFO,
            prefix: config.prefix ?? 'Ouroboros',
            enableTimestamp: config.enableTimestamp ?? true,
            enableColors: config.enableColors ?? true
        };
    }
    formatMessage(level, message) {
        const timestamp = this.config.enableTimestamp
            ? `[${new Date().toISOString()}] `
            : '';
        const prefix = this.config.prefix ? `[${this.config.prefix}] ` : '';
        const levelStr = `[${level}] `;
        return `${timestamp}${prefix}${levelStr}${message}`;
    }
    log(level, levelStr, message, ...args) {
        if (level < this.config.level)
            return;
        const formatted = this.formatMessage(levelStr, message);
        switch (level) {
            case LogLevel.DEBUG:
                console.debug(formatted, ...args);
                break;
            case LogLevel.INFO:
                console.info(formatted, ...args);
                break;
            case LogLevel.WARN:
                console.warn(formatted, ...args);
                break;
            case LogLevel.ERROR:
                console.error(formatted, ...args);
                break;
        }
    }
    debug(message, ...args) {
        this.log(LogLevel.DEBUG, 'DEBUG', message, ...args);
    }
    info(message, ...args) {
        this.log(LogLevel.INFO, 'INFO', message, ...args);
    }
    warn(message, ...args) {
        this.log(LogLevel.WARN, 'WARN', message, ...args);
    }
    error(message, ...args) {
        this.log(LogLevel.ERROR, 'ERROR', message, ...args);
    }
    setLevel(level) {
        this.config.level = level;
    }
    setPrefix(prefix) {
        this.config.prefix = prefix;
    }
}
export const logger = new Logger();
