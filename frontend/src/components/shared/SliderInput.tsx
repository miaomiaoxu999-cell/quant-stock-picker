"use client";

interface SliderInputProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  unit?: string;
  disabled?: boolean;
}

export default function SliderInput({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = "",
  disabled = false,
}: SliderInputProps) {
  return (
    <div className={disabled ? "opacity-50" : ""}>
      <div className="flex items-center justify-between mb-1">
        <label className="text-sm text-text-secondary">{label}</label>
        <span className="text-sm font-medium text-text-primary">
          {value}
          {unit && <span className="text-text-secondary ml-0.5">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full h-1.5 bg-border rounded-lg appearance-none cursor-pointer accent-blue disabled:cursor-not-allowed"
      />
      <div className="flex justify-between text-xs text-text-secondary mt-0.5">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}
