import { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Bot, User, AlertCircle, FileText, Video, Globe, Loader2, Database, Sparkles } from 'lucide-react';
import { api, ChatSource, RagStatsResponse } from '../services/api';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: ChatSource[];
    timestamp: Date;
    isLoading?: boolean;
    error?: string;
}

interface ChatViewProps {
    showNotification?: (message: string, type: 'info' | 'success' | 'error') => void;
}

export default function ChatView({ showNotification }: ChatViewProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [stats, setStats] = useState<RagStatsResponse | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            const data = await api.getRagStats();
            setStats(data);
        } catch (error) {
            console.error('Failed to fetch RAG stats:', error);
        }
    };

    const getSourceIcon = (source: string) => {
        switch (source.toLowerCase()) {
            case 'video':
                return <Video className="w-3 h-3" />;
            case 'doc':
                return <FileText className="w-3 h-3" />;
            case 'web':
                return <Globe className="w-3 h-3" />;
            default:
                return <FileText className="w-3 h-3" />;
        }
    };

    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        
        const query = input.trim();
        if (!query || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: query,
            timestamp: new Date(),
        };

        const assistantMessageId = (Date.now() + 1).toString();
        const loadingMessage: Message = {
            id: assistantMessageId,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isLoading: true,
        };

        setMessages(prev => [...prev, userMessage, loadingMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await api.chat(query);

            setMessages(prev => prev.map(msg => 
                msg.id === assistantMessageId
                    ? {
                        ...msg,
                        content: response.response || 'No response received.',
                        sources: response.sources,
                        isLoading: false,
                        error: response.status === 'error' ? response.message : undefined,
                    }
                    : msg
            ));

            if (response.status === 'error') {
                showNotification?.(response.message || 'Chat error', 'error');
            }
        } catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Failed to get response';
            setMessages(prev => prev.map(msg => 
                msg.id === assistantMessageId
                    ? {
                        ...msg,
                        content: 'Failed to get a response. Make sure Ollama is running.',
                        isLoading: false,
                        error: errorMsg,
                    }
                    : msg
            ));
            showNotification?.(errorMsg, 'error');
        } finally {
            setIsLoading(false);
            inputRef.current?.focus();
        }
    }, [input, isLoading, showNotification]);

    const renderMarkdown = (text: string) => {
        const lines = text.split('\n');
        const elements: JSX.Element[] = [];
        let listItems: string[] = [];

        const flushList = () => {
            if (listItems.length > 0) {
                elements.push(
                    <ul key={elements.length} className="list-disc list-inside space-y-1 my-2">
                        {listItems.map((item, i) => (
                            <li key={i} className="text-gray-300">{item}</li>
                        ))}
                    </ul>
                );
                listItems = [];
            }
        };

        lines.forEach((line, i) => {
            const trimmed = line.trim();

            if (trimmed.match(/^[-*•]\s/)) {
                listItems.push(trimmed.replace(/^[-*•]\s/, ''));
            } else if (trimmed.match(/^\d+\.\s/)) {
                listItems.push(trimmed.replace(/^\d+\.\s/, ''));
            } else {
                flushList();
                if (trimmed) {
                    const boldText = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    elements.push(
                        <p key={i} className="text-gray-300 my-1" dangerouslySetInnerHTML={{ __html: boldText }} />
                    );
                }
            }
        });

        flushList();
        return elements;
    };

    return (
        <div className="flex flex-col h-[calc(100vh-8rem)]">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-white flex items-center gap-2">
                        <Sparkles className="w-6 h-6 text-purple-400" />
                        Chat with your Notes
                    </h1>
                    <p className="text-gray-400 text-sm mt-1">
                        Ask questions about your captured content
                    </p>
                </div>
                
                {stats && stats.total_chunks !== undefined && (
                    <div className="flex items-center gap-2 text-sm text-gray-400 glass px-3 py-2 rounded-lg">
                        <Database className="w-4 h-4" />
                        <span>{stats.total_chunks} chunks indexed</span>
                        {stats.sessions && stats.sessions.length > 0 && (
                            <span className="text-gray-500">
                                from {stats.sessions.length} sessions
                            </span>
                        )}
                    </div>
                )}
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto glass rounded-xl p-4 mb-4 space-y-4">
                {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400">
                        <Bot className="w-16 h-16 mb-4 opacity-50" />
                        <p className="text-lg mb-2">No messages yet</p>
                        <p className="text-sm text-center max-w-md">
                            Ask questions about your captured videos, documents, and web articles.
                            The assistant will search your notes and provide relevant answers.
                        </p>
                        {stats?.total_chunks === 0 && (
                            <div className="mt-4 p-3 bg-yellow-500/10 border-[3px] border-yellow-500/30 rounded-lg text-yellow-400 text-sm">
                                <AlertCircle className="w-4 h-4 inline mr-2" />
                                No content indexed yet. Capture some content first!
                            </div>
                        )}
                    </div>
                ) : (
                    messages.map((message) => (
                        <div
                            key={message.id}
                            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                                    message.role === 'user'
                                        ? 'bg-purple-600/80 text-white ml-8'
                                        : 'glass-light text-gray-100 mr-8'
                                }`}
                            >
                                {/* Header with icon */}
                                <div className="flex items-center gap-2 mb-2">
                                    {message.role === 'user' ? (
                                        <User className="w-4 h-4" />
                                    ) : (
                                        <Bot className="w-4 h-4 text-purple-400" />
                                    )}
                                    <span className="text-xs text-gray-400">
                                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                </div>

                                {/* Content */}
                                {message.isLoading ? (
                                    <div className="flex items-center gap-2 text-gray-400">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span>Thinking...</span>
                                    </div>
                                ) : message.error ? (
                                    <div className="text-red-400">
                                        <AlertCircle className="w-4 h-4 inline mr-2" />
                                        {message.content}
                                    </div>
                                ) : (
                                    <div className="prose prose-invert prose-sm max-w-none">
                                        {renderMarkdown(message.content)}
                                    </div>
                                )}

                                {/* Sources */}
                                {message.sources && message.sources.length > 0 && (
                                    <div className="mt-3 pt-3 border-t-[3px] border-gray-600/50">
                                        <p className="text-xs text-gray-400 mb-2">Sources:</p>
                                        <div className="flex flex-wrap gap-2">
                                            {message.sources.map((source, i) => (
                                                <span
                                                    key={i}
                                                    className="inline-flex items-center gap-1 px-2 py-1 bg-gray-700/50 rounded-md text-xs text-gray-300"
                                                >
                                                    {getSourceIcon(source.source)}
                                                    <span className="truncate max-w-32" title={source.title}>
                                                        {source.title}
                                                    </span>
                                                    {source.relevance !== undefined && (
                                                        <span className="text-gray-500">
                                                            ({Math.round(source.relevance * 100)}%)
                                                        </span>
                                                    )}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <form onSubmit={handleSubmit} className="flex gap-3">
                <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about your notes..."
                    className="flex-1 glass px-4 py-3 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                    disabled={isLoading}
                />
                <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="px-6 py-3 rounded-xl text-white font-medium transition-all duration-200 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    style={{ background: '#ff6b6b' }}
                >
                    {isLoading ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                        <Send className="w-5 h-5" />
                    )}
                </button>
            </form>
        </div>
    );
}
