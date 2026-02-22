interface StepIndicatorProps {
  steps: string[];
  currentStep: number; // 1-based
  onStepClick?: (step: number) => void;
}

export default function StepIndicator({
  steps,
  currentStep,
  onStepClick,
}: StepIndicatorProps) {
  return (
    <div className="flex items-center justify-between w-full">
      {steps.map((label, i) => {
        const stepNum = i + 1;
        const isCompleted = stepNum < currentStep;
        const isActive = stepNum === currentStep;
        return (
          <div key={i} className="flex items-center flex-1 last:flex-none">
            {/* Circle + Label */}
            <div
              onClick={onStepClick ? () => onStepClick(stepNum) : undefined}
              className={`flex flex-col items-center ${
                onStepClick ? "cursor-pointer" : ""
              }`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
                  isCompleted
                    ? "bg-green text-white"
                    : isActive
                    ? "bg-blue text-white"
                    : "bg-bg-hover text-text-secondary border border-border"
                }`}
              >
                {isCompleted ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  stepNum
                )}
              </div>
              <span
                className={`text-xs mt-1 text-center whitespace-nowrap ${
                  isActive
                    ? "text-blue font-medium"
                    : isCompleted
                    ? "text-green"
                    : "text-text-secondary"
                }`}
              >
                {label}
              </span>
            </div>

            {/* Connecting line */}
            {i < steps.length - 1 && (
              <div
                className={`flex-1 h-0.5 mx-2 mt-[-1rem] ${
                  isCompleted ? "bg-green" : "bg-border"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
