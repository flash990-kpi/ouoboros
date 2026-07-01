export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  SILENT = 4
}

export interface LoggerConfig {
  level: LogLevel;
  prefix?: string;
  enableTimestamp?: boolean;
  enableColors?: boolean;
}

export class Logger {
  private config: LoggerConfig;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = {
      level: config.level ?? LogLevel.INFO,
      prefix: config.prefix ?? 'Ouroboros',
      enableTimestamp: config.enableTimestamp ?? true,
      enableColors: config.enableColors ?? true
    };
  }

  private formatMessage(level: string, message: string): string {
    const timestamp = this.config.enableTimestamp 
      ? `[${new Date().toISOString()}] ` 
      : '';
    const prefix = this.config.prefix ? `[${this.config.prefix}] ` : '';
    const levelStr = `[${level}] `;
    return `${timestamp}${prefix}${levelStr}${message}`;
  }

  private log(level: LogLevel, levelStr: string, message: string, ...args: any[]): void {
    if (level < this.config.level) return;

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

  debug(message: string, ...args: any[]): void {
    this.log(LogLevel.DEBUG, 'DEBUG', message, ...args);
  }

  info(message: string, ...args: any[]): void {
    this.log(LogLevel.INFO, 'INFO', message, ...args);
  }

  warn(message: string, ...args: any[]): void {
    this.log(LogLevel.WARN, 'WARN', message, ...args);
  }

  error(message: string, ...args: any[]): void {
    this.log(LogLevel.ERROR, 'ERROR', message, ...args);
  }

  setLevel(level: LogLevel): void {
    this.config.level = level;
  }

  setPrefix(prefix: string): void {
    this.config.prefix = prefix;
  }
}

export const logger = new Logger();
