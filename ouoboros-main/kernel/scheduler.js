export class Scheduler {
    constructor() {
        this.tasksQueue = [];
        this.executionLock = false;
    }
    enqueue(task) {
        const fullTask = { ...task, timestamp: performance.now() };
        this.tasksQueue.push(fullTask);
        // Algoritmo di ordinamento: priorità decrescente, poi ordine cronologico d'arrivo
        this.tasksQueue.sort((nodeA, nodeB) => {
            if (nodeB.priority === nodeA.priority) {
                return nodeA.timestamp - nodeB.timestamp;
            }
            return nodeB.priority - nodeA.priority;
        });
        this.processQueue();
    }
    async processQueue() {
        if (this.executionLock || this.tasksQueue.length === 0) {
            return;
        }
        this.executionLock = true;
        const currentActiveTask = this.tasksQueue.shift();
        try {
            await currentActiveTask.action();
        }
        catch (err) {
            console.error(`[CRITICAL KERNEL SCHEDULER FAULT] Errore nell'esecuzione del task ${currentActiveTask.id}:`, err);
        }
        finally {
            this.executionLock = false;
            // Richiamiamo in differita l'esecuzione per liberare lo stack di chiamate
            setTimeout(() => this.processQueue(), 0);
        }
    }
    clear() {
        this.tasksQueue = [];
    }
}
