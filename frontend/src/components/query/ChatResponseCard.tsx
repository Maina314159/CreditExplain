import React from 'react';
import { CheckCircle, FileText, Zap } from 'lucide-react';
import { type QueryResponse } from '@/types/api.types';

interface ChatResponseCardProps {
    setInput: (input: string) => void;
    response: QueryResponse;
}

const ChatResponseCard: React.FC<ChatResponseCardProps> = ({ response, setInput }) => {

    return (
        <div className="max-w-[90%] mx-auto bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 rounded-2xl p-6 border border-blue-200 dark:border-blue-800 shadow-sm animate-in slide-in-from-bottom-8 duration-700">
            <div className="flex items-center mb-5">
                <div className="flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                    <span className="font-semibold text-slate-800 dark:text-slate-200">Analysis Complete</span>
                </div>
                <div className="ml-auto flex items-center space-x-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Confidence:</span>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium shadow-sm ${
                        response.confidence === 'HIGH' 
                            ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200' 
                            : response.confidence === 'MEDIUM' 
                            ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
                            : 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
                    }`}>
                        {response.confidence}
                    </div>
                </div>
            </div>

            {/* Citations */}
            {response.citations.length > 0 && (
                <div className="mb-5">
                    <h4 className="font-semibold text-slate-800 dark:text-slate-200 mb-3 flex items-center gap-2">
                        <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                        Source Documents
                    </h4>
                    <div className="space-y-3">
                        {response.citations.map((citation, idx) => (
                            <div key={idx} className="bg-white dark:bg-slate-950 p-4 rounded-xl border border-slate-200/60 dark:border-slate-800/60 shadow-sm hover:shadow-md transition-shadow">
                                <div className="flex items-center mb-2">
                                    <span className="font-medium text-sm text-blue-700 dark:text-blue-300">
                                        {citation.doc_id}
                                    </span>
                                    {citation.chunk_id && (
                                        <span className="ml-3 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-2.5 py-0.5 rounded-full text-xs font-medium">
                                            Chunk {citation.chunk_id}
                                        </span>
                                    )}
                                </div>
                                <p className="font-serif text-sm text-slate-600 dark:text-slate-400 italic">
                                    "{citation.text_excerpt}"
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Follow-up Questions */}
            {response.follow_up_questions && response.follow_up_questions.length > 0 && (
                <div>
                    <h4 className="font-semibold text-slate-800 dark:text-slate-200 mb-3 flex items-center gap-2">
                        <Zap className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                        Related Questions
                    </h4>
                    <div className="flex flex-wrap gap-2">
                        {response.follow_up_questions.map((question, idx) => (
                            <button
                                key={idx}
                                onClick={() => setInput(question)}
                                className="bg-white dark:bg-slate-950 hover:bg-blue-50 dark:hover:bg-blue-950 border border-slate-200/60 dark:border-slate-800/60 rounded-full px-4 py-2 text-sm text-slate-700 dark:text-slate-300 transition-all duration-200 hover:border-blue-300 dark:hover:border-blue-700 hover:shadow-sm"
                            >
                                {question}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default ChatResponseCard;